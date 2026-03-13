"""
ValidationEngine for Signal Stability Gateway

Computes per-channel quality metrics using fully vectorized operations:
- SNR (Signal-to-Noise Ratio) via rolling EMA
- ISI Violations via np.lexsort + np.diff + np.bincount
- Firing Rate for TN-VAE latent space
- Viability Mask combining all criteria

NO PYTHON LOOPS over channels. All operations are O(spikes) or O(channels).
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from ..core.constants import (
    N_CHANNELS,
    SAMPLE_RATE_HZ,
    SNR_THRESHOLD,
    ISI_REFRACTORY_PERIOD_US,
    ISI_VIOLATION_LIMIT,
    IMPEDANCE_MIN_KOHM,
    IMPEDANCE_MAX_KOHM,
    SPIKE_DETECTION_THRESHOLD_MAD,
    MAD_TO_STD_FACTOR,
    EMA_ALPHA,
    VALIDATION_WINDOW_SEC,
    MAX_SPIKES_IN_WINDOW,
)
from ..core.data_types import ChannelMetrics, SanitizedFrame


@dataclass
class SpikeBuffer:
    """
    Ring buffer for spike events with vectorized storage.

    Uses structured arrays for efficient sorting and grouping.
    """
    channels: np.ndarray  # (capacity,), uint16
    timestamps: np.ndarray  # (capacity,), uint64
    amplitudes: np.ndarray  # (capacity,), float32
    head: int = 0
    count: int = 0
    capacity: int = MAX_SPIKES_IN_WINDOW

    @classmethod
    def create(cls, capacity: int = MAX_SPIKES_IN_WINDOW) -> 'SpikeBuffer':
        """Create empty spike buffer."""
        return cls(
            channels=np.zeros(capacity, dtype=np.uint16),
            timestamps=np.zeros(capacity, dtype=np.uint64),
            amplitudes=np.zeros(capacity, dtype=np.float32),
            capacity=capacity,
        )

    def add_spikes(
        self,
        channels: np.ndarray,
        timestamps: np.ndarray,
        amplitudes: np.ndarray,
    ) -> None:
        """Add multiple spikes (vectorized)."""
        n = len(channels)
        if n == 0:
            return

        # Handle wraparound
        end = self.head + n
        if end <= self.capacity:
            self.channels[self.head:end] = channels
            self.timestamps[self.head:end] = timestamps
            self.amplitudes[self.head:end] = amplitudes
        else:
            # Wraparound
            first_part = self.capacity - self.head
            self.channels[self.head:] = channels[:first_part]
            self.timestamps[self.head:] = timestamps[:first_part]
            self.amplitudes[self.head:] = amplitudes[:first_part]

            second_part = n - first_part
            self.channels[:second_part] = channels[first_part:]
            self.timestamps[:second_part] = timestamps[first_part:]
            self.amplitudes[:second_part] = amplitudes[first_part:]

        self.head = end % self.capacity
        self.count = min(self.count + n, self.capacity)

    def get_valid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get all valid spikes (unordered)."""
        if self.count < self.capacity:
            return (
                self.channels[:self.count].copy(),
                self.timestamps[:self.count].copy(),
                self.amplitudes[:self.count].copy(),
            )
        return (
            self.channels.copy(),
            self.timestamps.copy(),
            self.amplitudes.copy(),
        )

    def clear(self) -> None:
        """Reset buffer."""
        self.head = 0
        self.count = 0


class ValidationEngine:
    """
    Computes per-channel quality metrics for viability scoring.

    Uses rolling EMA for SNR (no large matrix operations).
    Uses vectorized spike analysis for ISI violations.

    FDA Documentation:
        - SNR Threshold: >= 4.0 (SpikeAgent benchmark)
        - ISI Violation Limit: < 1.5% (biophysical refractory period)
        - Impedance Range: 50-3000 kOhm (clinical electrode spec)
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        sample_rate_hz: int = SAMPLE_RATE_HZ,
    ):
        """
        Initialize ValidationEngine.

        Args:
            n_channels: Number of recording channels
            sample_rate_hz: Sampling rate in Hz
        """
        self.n_channels = n_channels
        self.sample_rate = sample_rate_hz

        # Rolling statistics for SNR (EMA-based)
        # Noise estimate: MAD-based, updated per batch
        self._ema_noise = np.ones(n_channels, dtype=np.float32) * 10.0
        # Signal estimate: 95th percentile amplitude, EMA-smoothed
        self._ema_signal = np.zeros(n_channels, dtype=np.float32)

        # Spike buffer for ISI analysis
        self._spike_buffer = SpikeBuffer.create()

        # Spike counts per channel (for firing rate)
        self._spike_counts = np.zeros(n_channels, dtype=np.int64)
        self._total_duration_sec = 0.0

        # Impedance values (updated externally)
        self._impedance_kohm = np.full(n_channels, 1000.0, dtype=np.float32)

        # Batch counter
        self._batch_count = 0

    def process(
        self,
        sanitized: SanitizedFrame,
        timestamps: np.ndarray,
    ) -> ChannelMetrics:
        """
        Process sanitized frame and compute channel metrics.

        Pipeline:
            1. Detect spikes in spike band
            2. Update rolling SNR (EMA)
            3. Compute ISI violations (vectorized)
            4. Compute firing rates
            5. Generate viability mask

        Args:
            sanitized: SanitizedFrame from SanitizationLayer
            timestamps: Timestamps for this batch

        Returns:
            ChannelMetrics with per-channel quality scores
        """
        batch_size = sanitized.spikes.shape[0]
        batch_duration_sec = batch_size / self.sample_rate

        # Step 1: Detect spikes
        spike_channels, spike_times, spike_amps = self._detect_spikes(
            sanitized.spikes, timestamps
        )

        # Add spikes to buffer
        self._spike_buffer.add_spikes(spike_channels, spike_times, spike_amps)

        # Update spike counts per channel
        if len(spike_channels) > 0:
            np.add.at(self._spike_counts, spike_channels, 1)

        self._total_duration_sec += batch_duration_sec

        # Step 2: Update rolling SNR
        snr = self._compute_snr(sanitized.spikes)

        # Step 3: Compute ISI violations (vectorized)
        isi_violation_rate = self._compute_isi_violations()

        # Step 4: Compute firing rate
        firing_rate_hz = self._compute_firing_rate()

        # Step 5: Generate viability mask
        viability_mask = self._compute_viability_mask(
            snr, isi_violation_rate, sanitized.artifact_flags
        )

        self._batch_count += 1

        return ChannelMetrics(
            timestamp_us=sanitized.timestamp_us,
            snr=snr,
            firing_rate_hz=firing_rate_hz,
            isi_violation_rate=isi_violation_rate,
            impedance_kohm=self._impedance_kohm.copy(),
            viability_mask=viability_mask,
            viable_channel_count=int(viability_mask.sum()),
        )

    def _detect_spikes(
        self,
        spike_band: np.ndarray,
        timestamps: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect spikes using MAD-based threshold with local minima + refractory period.

        VECTORIZED: Uses boolean indexing with peak detection.
        Only counts local minima that cross threshold (actual spike peaks).
        Enforces refractory period to avoid counting noise as multiple spikes.

        Args:
            spike_band: Bandpass filtered data, shape (batch_size, n_channels)
            timestamps: Microsecond timestamps, shape (batch_size,)

        Returns:
            Tuple of (channel_ids, spike_timestamps, amplitudes)
        """
        batch_size = spike_band.shape[0]

        if batch_size < 3:
            return (
                np.array([], dtype=np.uint16),
                np.array([], dtype=np.uint64),
                np.array([], dtype=np.float32),
            )

        # Compute MAD per channel (robust to outliers)
        median = np.median(spike_band, axis=0)
        mad = np.median(np.abs(spike_band - median), axis=0)
        mad = np.maximum(mad, 1e-6)  # Prevent division by zero

        # Threshold: negative for extracellular spikes
        threshold = median + SPIKE_DETECTION_THRESHOLD_MAD * mad * MAD_TO_STD_FACTOR

        # Find LOCAL MINIMA (actual spike peaks for extracellular recordings)
        # A local minimum: sample[i] < sample[i-1] AND sample[i] < sample[i+1]
        is_local_min = (
            (spike_band[1:-1] < spike_band[:-2]) &
            (spike_band[1:-1] < spike_band[2:])
        )

        # Also must be below threshold
        below_threshold = spike_band[1:-1] < threshold

        # Combine: local minimum AND below threshold
        is_spike = is_local_min & below_threshold

        # Get indices (offset by 1 because we excluded first sample)
        time_idx, channel_idx = np.where(is_spike)
        time_idx = time_idx + 1  # Adjust for slicing offset

        if len(time_idx) == 0:
            return (
                np.array([], dtype=np.uint16),
                np.array([], dtype=np.uint64),
                np.array([], dtype=np.float32),
            )

        # REFRACTORY PERIOD ENFORCEMENT
        # Only keep the first spike within each refractory window per channel
        # Refractory period in samples (1.5ms at 20kHz = 30 samples)
        refractory_samples = int(ISI_REFRACTORY_PERIOD_US * self.sample_rate / 1_000_000)

        # Sort by channel then time
        sort_idx = np.lexsort((time_idx, channel_idx))
        time_idx = time_idx[sort_idx]
        channel_idx = channel_idx[sort_idx]

        # Find valid spikes (respecting refractory period)
        keep = np.ones(len(time_idx), dtype=bool)

        # Use vectorized approach: check if enough time passed since last spike on same channel
        if len(time_idx) > 1:
            # Time difference and channel difference between consecutive spikes
            time_diff = np.diff(time_idx)
            chan_diff = np.diff(channel_idx)

            # Invalid if same channel AND time diff < refractory
            invalid = (chan_diff == 0) & (time_diff < refractory_samples)
            keep[1:] = ~invalid

            # Apply keep mask
            time_idx = time_idx[keep]
            channel_idx = channel_idx[keep]

        # Get timestamps and amplitudes
        spike_timestamps = timestamps[time_idx]
        spike_amplitudes = spike_band[time_idx, channel_idx]

        return (
            channel_idx.astype(np.uint16),
            spike_timestamps.astype(np.uint64),
            spike_amplitudes.astype(np.float32),
        )

    def _compute_snr(self, spike_band: np.ndarray) -> np.ndarray:
        """
        Compute SNR using rolling EMA (NO large matrix operations).

        Noise: MAD-based estimate, converted to std
        Signal: 95th percentile amplitude per channel

        Args:
            spike_band: Bandpass filtered data, shape (batch_size, n_channels)

        Returns:
            SNR per channel, shape (n_channels,)
        """
        # Compute batch noise estimate (MAD-based)
        median = np.median(spike_band, axis=0)
        mad = np.median(np.abs(spike_band - median), axis=0)
        batch_noise = mad * MAD_TO_STD_FACTOR
        batch_noise = np.maximum(batch_noise, 1e-6)

        # Compute batch signal estimate (95th percentile of amplitude)
        batch_signal = np.percentile(np.abs(spike_band), 95, axis=0)

        # Update EMA estimates
        if self._batch_count == 0:
            self._ema_noise = batch_noise.astype(np.float32)
            self._ema_signal = batch_signal.astype(np.float32)
        else:
            self._ema_noise = (
                EMA_ALPHA * batch_noise + (1 - EMA_ALPHA) * self._ema_noise
            ).astype(np.float32)
            self._ema_signal = (
                EMA_ALPHA * batch_signal + (1 - EMA_ALPHA) * self._ema_signal
            ).astype(np.float32)

        # Compute SNR
        snr = self._ema_signal / self._ema_noise

        return snr.astype(np.float32)

    def _compute_isi_violations(self) -> np.ndarray:
        """
        Compute ISI violation rate using FULLY VECTORIZED operations.

        Algorithm:
            1. Sort spikes by (channel, timestamp) using np.lexsort
            2. Compute ISI via np.diff
            3. Detect channel boundaries
            4. Count violations with np.bincount

        NO PYTHON LOOPS over channels.

        Returns:
            ISI violation rate per channel, shape (n_channels,)
        """
        channels, timestamps, _ = self._spike_buffer.get_valid()

        if len(channels) == 0:
            return np.zeros(self.n_channels, dtype=np.float32)

        # Sort by (channel, timestamp) - lexsort uses last key first
        sort_idx = np.lexsort((timestamps, channels))
        sorted_channels = channels[sort_idx]
        sorted_timestamps = timestamps[sort_idx]

        # Compute ISI (time difference between consecutive spikes)
        isi = np.diff(sorted_timestamps).astype(np.int64)

        # Detect channel boundaries (where channel changes)
        channel_diff = np.diff(sorted_channels)
        same_channel = channel_diff == 0

        # ISI violations: same channel AND ISI < refractory period
        violations = same_channel & (isi < ISI_REFRACTORY_PERIOD_US)

        # Count violations per channel using bincount
        # Channel ID for each ISI is sorted_channels[:-1] (first spike of pair)
        violation_channels = sorted_channels[:-1][violations]
        violation_counts = np.bincount(
            violation_channels,
            minlength=self.n_channels
        ).astype(np.float32)

        # Count total spikes per channel
        spike_counts = np.bincount(
            channels,
            minlength=self.n_channels
        ).astype(np.float32)

        # Violation rate = violations / (spikes - 1)
        # Avoid division by zero
        spike_pairs = np.maximum(spike_counts - 1, 1)
        violation_rate = violation_counts / spike_pairs

        return violation_rate

    def _compute_firing_rate(self) -> np.ndarray:
        """
        Compute firing rate per channel using bincount.

        O(n_spikes) operation, not O(n_channels).

        Returns:
            Firing rate in Hz per channel, shape (n_channels,)
        """
        if self._total_duration_sec <= 0:
            return np.zeros(self.n_channels, dtype=np.float32)

        firing_rate = self._spike_counts / self._total_duration_sec

        return firing_rate.astype(np.float32)

    def _compute_viability_mask(
        self,
        snr: np.ndarray,
        isi_violation_rate: np.ndarray,
        artifact_flags: np.ndarray,
    ) -> np.ndarray:
        """
        Compute channel viability mask.

        Criteria:
            - SNR >= 4.0
            - ISI violation rate < 1.5%
            - Impedance in 50-3000 kOhm range
            - No artifact flags

        Args:
            snr: SNR per channel
            isi_violation_rate: ISI violation rate per channel
            artifact_flags: Artifact flags per channel

        Returns:
            Boolean viability mask, shape (n_channels,)
        """
        viability_mask = (
            (snr >= SNR_THRESHOLD) &
            (isi_violation_rate < ISI_VIOLATION_LIMIT) &
            (self._impedance_kohm >= IMPEDANCE_MIN_KOHM) &
            (self._impedance_kohm <= IMPEDANCE_MAX_KOHM) &
            (~artifact_flags)
        )

        return viability_mask

    def update_impedance(self, impedance_kohm: np.ndarray) -> None:
        """
        Update electrode impedance values.

        Called periodically by external impedance measurement system.

        Args:
            impedance_kohm: Impedance per channel in kOhms
        """
        assert impedance_kohm.shape == (self.n_channels,), \
            f"Expected shape ({self.n_channels},), got {impedance_kohm.shape}"
        self._impedance_kohm = impedance_kohm.astype(np.float32)

    def get_region_metrics(
        self,
        metrics: ChannelMetrics,
        start_ch: int,
        end_ch: int,
    ) -> dict:
        """
        Get aggregated metrics for a channel region.

        Args:
            metrics: ChannelMetrics from process()
            start_ch: Start channel index (inclusive)
            end_ch: End channel index (exclusive)

        Returns:
            Dict with region statistics
        """
        region_mask = metrics.viability_mask[start_ch:end_ch]
        region_snr = metrics.snr[start_ch:end_ch]
        region_fr = metrics.firing_rate_hz[start_ch:end_ch]

        return {
            'viable_count': int(region_mask.sum()),
            'total_count': end_ch - start_ch,
            'viability_pct': 100.0 * region_mask.sum() / (end_ch - start_ch),
            'mean_snr': float(np.mean(region_snr)),
            'mean_firing_rate_hz': float(np.mean(region_fr)),
        }

    def reset(self) -> None:
        """Reset all statistics."""
        self._ema_noise.fill(10.0)
        self._ema_signal.fill(0.0)
        self._spike_buffer.clear()
        self._spike_counts.fill(0)
        self._total_duration_sec = 0.0
        self._batch_count = 0
