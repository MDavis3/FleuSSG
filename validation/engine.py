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

from ..core.array_types import BoolVector, FloatMatrix, FloatVector, TimestampVector
from ..core.constants import (
    N_CHANNELS,
    SAMPLE_RATE_HZ,
    SNR_THRESHOLD,
    ISI_VIOLATION_LIMIT,
    IMPEDANCE_MIN_KOHM,
    IMPEDANCE_MAX_KOHM,
    MAD_TO_STD_FACTOR,
    EMA_ALPHA,
    VALIDATION_WINDOW_SEC,
)
from ..core.data_types import ChannelMetrics, RegionMetrics, SanitizedFrame
from .spike_analysis import (
    SpikeBuffer,
    compute_firing_rate,
    compute_isi_violations,
    detect_spikes,
)


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

        # Impedance values (updated externally)
        self._impedance_kohm = np.full(n_channels, 1000.0, dtype=np.float32)

        # Batch counter
        self._batch_count = 0
        self._observed_duration_sec = 0.0

    def process(
        self,
        sanitized: SanitizedFrame,
        timestamps: TimestampVector,
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
        spike_channels, spike_times, spike_amps = detect_spikes(
            sanitized.spikes,
            timestamps,
            self.sample_rate,
        )

        # Add spikes to buffer
        self._spike_buffer.add_spikes(spike_channels, spike_times, spike_amps)
        if len(timestamps) > 0:
            window_start_us = int(timestamps[-1]) - int(VALIDATION_WINDOW_SEC * 1_000_000)
            self._spike_buffer.trim_before(window_start_us)
        self._observed_duration_sec = min(
            self._observed_duration_sec + batch_duration_sec,
            VALIDATION_WINDOW_SEC,
        )

        # Step 2: Update rolling SNR
        snr = self._compute_snr(sanitized.spikes)

        # Step 3: Compute ISI violations (vectorized)
        isi_violation_rate = compute_isi_violations(
            self._spike_buffer,
            self.n_channels,
        )

        # Step 4: Compute firing rate
        firing_rate_hz = compute_firing_rate(
            self._spike_buffer,
            self.n_channels,
            self._observed_duration_sec,
        )

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

    def _compute_snr(self, spike_band: FloatMatrix) -> FloatVector:
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

    def _compute_viability_mask(
        self,
        snr: FloatVector,
        isi_violation_rate: FloatVector,
        artifact_flags: BoolVector,
    ) -> BoolVector:
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

    def update_impedance(self, impedance_kohm: FloatVector) -> None:
        """
        Update electrode impedance values.

        Called periodically by external impedance measurement system.

        Args:
            impedance_kohm: Impedance per channel in kOhms
        """
        if impedance_kohm.shape != (self.n_channels,):
            raise ValueError(
                f"Expected shape ({self.n_channels},), got {impedance_kohm.shape}"
            )
        self._impedance_kohm = impedance_kohm.astype(np.float32)

    def get_region_metrics(
        self,
        metrics: ChannelMetrics,
        start_ch: int,
        end_ch: int,
    ) -> RegionMetrics:
        """
        Get aggregated metrics for a channel region.

        Args:
            metrics: ChannelMetrics from process()
            start_ch: Start channel index (inclusive)
            end_ch: End channel index (exclusive)

        Returns:
            Aggregated region statistics
        """
        region_mask = metrics.viability_mask[start_ch:end_ch]
        region_snr = metrics.snr[start_ch:end_ch]
        region_fr = metrics.firing_rate_hz[start_ch:end_ch]

        total_count = end_ch - start_ch
        return RegionMetrics(
            viable_count=int(region_mask.sum()),
            total_count=total_count,
            viability_pct=100.0 * region_mask.sum() / total_count,
            mean_snr=float(np.mean(region_snr)),
            mean_firing_rate_hz=float(np.mean(region_fr)),
        )

    def reset(self) -> None:
        """Reset all statistics."""
        self._ema_noise.fill(10.0)
        self._ema_signal.fill(0.0)
        self._spike_buffer.clear()
        self._observed_duration_sec = 0.0
        self._batch_count = 0
