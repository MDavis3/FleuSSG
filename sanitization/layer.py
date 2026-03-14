"""
SanitizationLayer for Signal Stability Gateway

DSP pipeline for 1024-channel neural data:
1. 60Hz notch filter (+ harmonics)
2. LFP extraction (lowpass <300Hz)
3. Spike band extraction (bandpass 300-3000Hz)
4. Artifact rejection (Tanh-scaling with rolling sigma)

OPTIMIZED: Uses scipy.signal.sosfilt for vectorized, numerically stable filtering.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional
from dataclasses import dataclass

from ..core.constants import (
    N_CHANNELS,
    SAMPLE_RATE_HZ,
    BUTTERWORTH_ORDER,
    LFP_CUTOFF_HZ,
    SPIKE_LOWCUT_HZ,
    SPIKE_HIGHCUT_HZ,
    NOTCH_FREQUENCIES_HZ,
    NOTCH_QUALITY_FACTOR,
    ARTIFACT_THRESHOLD_SIGMA,
    EMA_ALPHA,
)
from ..core.data_types import SanitizedFrame


@dataclass
class SOSFilterState:
    """
    Maintains SOS filter states for continuous processing.

    Uses Second-Order Sections (SOS) format for numerical stability.
    """
    notch_zi: list  # List of zi states for each notch filter
    lfp_zi: np.ndarray  # Lowpass filter state
    spike_zi: np.ndarray  # Bandpass filter state


class SanitizationLayer:
    """
    DSP pipeline for 1024-channel neural signals.

    OPTIMIZED: Uses sosfilt for all filtering operations.
    - sosfilt is 2-5x faster than lfilter for high-order filters
    - SOS format is numerically stable for cascaded biquads
    - Vectorized across all channels simultaneously

    FDA Documentation:
        - Notch: IIR notch at 60, 120, 180 Hz (Q=30)
        - LFP: 4th-order Butterworth lowpass, fc=300Hz
        - Spike: 4th-order Butterworth bandpass, 300-3000Hz
        - Artifact: Tanh-scaling at 4σ threshold
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        sample_rate_hz: int = SAMPLE_RATE_HZ,
    ):
        """
        Initialize SanitizationLayer with SOS filters.

        Pre-computes all filter coefficients in SOS format at initialization.

        Args:
            n_channels: Number of recording channels
            sample_rate_hz: Sampling rate in Hz
        """
        self.n_channels = n_channels
        self.sample_rate = sample_rate_hz
        self.nyquist = sample_rate_hz / 2.0

        # Pre-compute filter coefficients in SOS format
        self._notch_sos_list = self._design_notch_filters_sos()
        self._lfp_sos = self._design_lfp_filter_sos()
        self._spike_sos = self._design_spike_filter_sos()

        # Initialize filter states (will be set on first call)
        self._filter_state: Optional[SOSFilterState] = None

        # EMA-based rolling sigma for artifact rejection
        # Initialize high to avoid false positives during warmup
        self._ema_sigma = np.ones(n_channels, dtype=np.float32) * 50.0
        self._rolling_count = 0

        # Batch counter
        self._batch_id = 0

    def _design_notch_filters_sos(self) -> list:
        """
        Design IIR notch filters for 60Hz and harmonics in SOS format.

        Returns:
            List of SOS arrays for each notch frequency.
        """
        sos_list = []
        for freq in NOTCH_FREQUENCIES_HZ:
            if freq >= self.nyquist:
                continue
            # iirnotch returns b, a - convert to SOS
            b, a = signal.iirnotch(freq, NOTCH_QUALITY_FACTOR, self.sample_rate)
            # For 2nd-order filter, create single SOS section
            sos = np.array([[b[0], b[1], b[2], a[0], a[1], a[2]]])
            sos_list.append(sos)
        return sos_list

    def _design_lfp_filter_sos(self) -> np.ndarray:
        """
        Design 4th-order Butterworth lowpass for LFP extraction in SOS format.

        Returns:
            SOS array for the filter
        """
        wn = LFP_CUTOFF_HZ / self.nyquist
        sos = signal.butter(BUTTERWORTH_ORDER, wn, btype='low', output='sos')
        return sos

    def _design_spike_filter_sos(self) -> np.ndarray:
        """
        Design 4th-order Butterworth bandpass for spike extraction in SOS format.

        Returns:
            SOS array for the filter
        """
        low = SPIKE_LOWCUT_HZ / self.nyquist
        high = SPIKE_HIGHCUT_HZ / self.nyquist
        sos = signal.butter(BUTTERWORTH_ORDER, [low, high], btype='band', output='sos')
        return sos

    def _init_filter_states(self) -> None:
        """
        Initialize SOS filter states for streaming operation.

        Uses scipy.signal.sosfilt_zi for proper initial conditions.
        """
        # Notch filter states
        notch_zi = []
        for sos in self._notch_sos_list:
            zi = signal.sosfilt_zi(sos)
            # Expand for all channels: shape (n_sections, 2, n_channels)
            zi_expanded = np.tile(zi[:, :, np.newaxis], (1, 1, self.n_channels))
            notch_zi.append(zi_expanded.copy())

        # LFP filter state
        lfp_zi = signal.sosfilt_zi(self._lfp_sos)
        lfp_zi = np.tile(lfp_zi[:, :, np.newaxis], (1, 1, self.n_channels))

        # Spike filter state
        spike_zi = signal.sosfilt_zi(self._spike_sos)
        spike_zi = np.tile(spike_zi[:, :, np.newaxis], (1, 1, self.n_channels))

        self._filter_state = SOSFilterState(
            notch_zi=notch_zi,
            lfp_zi=lfp_zi.copy(),
            spike_zi=spike_zi.copy(),
        )

    def process(
        self,
        samples: np.ndarray,
        timestamps: np.ndarray,
    ) -> SanitizedFrame:
        """
        Process a batch of samples through the sanitization pipeline.

        OPTIMIZED: Uses sosfilt for all filtering - 2-5x faster than lfilter.

        Pipeline:
            1. Apply notch filters (60Hz + harmonics)
            2. Extract LFP band (<300Hz)
            3. Extract spike band (300-3000Hz)
            4. Detect artifacts using rolling sigma

        Args:
            samples: Raw voltage data, shape (batch_size, n_channels), float32 µV
            timestamps: Microsecond timestamps, shape (batch_size,), uint64

        Returns:
            SanitizedFrame with filtered signals and artifact flags
        """
        # Initialize filter states on first call
        if self._filter_state is None:
            self._init_filter_states()

        # Store raw for passthrough
        raw_unfiltered = samples.copy()

        # Step 1: Apply notch filters (cascaded) using sosfilt
        notched = samples.astype(np.float64)
        for i, sos in enumerate(self._notch_sos_list):
            notched, self._filter_state.notch_zi[i] = signal.sosfilt(
                sos, notched, axis=0, zi=self._filter_state.notch_zi[i]
            )

        # Step 2: Extract LFP band (lowpass) using sosfilt
        lfp, self._filter_state.lfp_zi = signal.sosfilt(
            self._lfp_sos, notched, axis=0,
            zi=self._filter_state.lfp_zi
        )

        # Step 3: Extract spike band (bandpass) using sosfilt
        spikes, self._filter_state.spike_zi = signal.sosfilt(
            self._spike_sos, notched, axis=0,
            zi=self._filter_state.spike_zi
        )

        # Step 4: Update rolling sigma and detect artifacts
        artifact_flags = self._detect_artifacts(notched)

        # Increment batch counter
        self._batch_id += 1

        return SanitizedFrame(
            timestamp_us=int(timestamps[0]) if len(timestamps) > 0 else 0,
            raw_unfiltered=raw_unfiltered.astype(np.float32),
            lfp=lfp.astype(np.float32),
            spikes=spikes.astype(np.float32),
            artifact_flags=artifact_flags,
        )

    def _detect_artifacts(self, data: np.ndarray) -> np.ndarray:
        """
        Detect artifacts using Tanh-scaling with rolling sigma.

        Uses EMA for efficient rolling sigma computation.

        Args:
            data: Notch-filtered data, shape (batch_size, n_channels)

        Returns:
            Boolean artifact flags, shape (n_channels,)
        """
        # Compute batch statistics per channel (vectorized)
        batch_std = np.std(data, axis=0, dtype=np.float64)
        batch_max = np.max(np.abs(data), axis=0)

        # Update rolling sigma with EMA
        if self._rolling_count == 0:
            self._ema_sigma = batch_std.astype(np.float32)
        else:
            self._ema_sigma = (
                EMA_ALPHA * batch_std + (1 - EMA_ALPHA) * self._ema_sigma
            ).astype(np.float32)

        self._rolling_count += data.shape[0]

        # Flag channels where max amplitude exceeds threshold
        threshold = ARTIFACT_THRESHOLD_SIGMA * self._ema_sigma
        artifact_flags = batch_max > threshold

        return artifact_flags.astype(bool)

    def apply_tanh_scaling(
        self,
        data: np.ndarray,
        artifact_flags: np.ndarray,
    ) -> np.ndarray:
        """
        Apply Tanh-scaling to attenuate artifacts while preserving signal shape.

        VECTORIZED: No Python loops over channels.

        Args:
            data: Input data, shape (batch_size, n_channels)
            artifact_flags: Per-channel artifact flags, shape (n_channels,)

        Returns:
            Scaled data with artifacts attenuated
        """
        if not np.any(artifact_flags):
            return data

        result = data.copy()

        # Vectorized tanh scaling for flagged channels
        threshold = ARTIFACT_THRESHOLD_SIGMA * self._ema_sigma
        threshold = np.maximum(threshold, 1e-6)  # Prevent division by zero

        # Apply to flagged channels only (vectorized)
        flagged = artifact_flags
        if flagged.any():
            normalized = result[:, flagged] / threshold[flagged]
            result[:, flagged] = threshold[flagged] * np.tanh(normalized)

        return result

    def get_rolling_sigma(self) -> np.ndarray:
        """Get current rolling sigma estimates per channel."""
        return self._ema_sigma.copy()

    def reset(self) -> None:
        """Reset all filter states and rolling statistics."""
        self._filter_state = None
        self._rolling_count = 0
        self._ema_sigma.fill(10.0)
        self._batch_id = 0
