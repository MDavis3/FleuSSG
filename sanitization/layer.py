"""
SanitizationLayer for Signal Stability Gateway

DSP pipeline for 1024-channel neural data:
1. 60Hz notch filter (+ harmonics)
2. LFP extraction (lowpass <300Hz)
3. Spike band extraction (bandpass 300-3000Hz)
4. Artifact rejection (Tanh-scaling with rolling sigma)

All operations are vectorized using NumPy/SciPy.
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
    SIGMA_WINDOW_SEC,
    EMA_ALPHA,
)
from ..core.data_types import SanitizedFrame


@dataclass
class FilterState:
    """
    Maintains IIR filter states for continuous processing.

    Prevents transient artifacts at batch boundaries.
    """
    notch_zi: list  # List of zi states for each notch filter
    lfp_zi: np.ndarray  # Lowpass filter state
    spike_zi: np.ndarray  # Bandpass filter state


class SanitizationLayer:
    """
    DSP pipeline for 1024-channel neural signals.

    Filters are designed for streaming (maintains state across batches).
    Artifact rejection uses rolling sigma from 10-second window.

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
        Initialize SanitizationLayer.

        Pre-computes all filter coefficients at initialization.

        Args:
            n_channels: Number of recording channels
            sample_rate_hz: Sampling rate in Hz
        """
        self.n_channels = n_channels
        self.sample_rate = sample_rate_hz
        self.nyquist = sample_rate_hz / 2.0

        # Pre-compute filter coefficients
        self._notch_coeffs = self._design_notch_filters()
        self._lfp_b, self._lfp_a = self._design_lfp_filter()
        self._spike_b, self._spike_a = self._design_spike_filter()

        # Initialize filter states (will be set on first call)
        self._filter_state: Optional[FilterState] = None

        # Rolling sigma for artifact rejection (Welford's algorithm)
        # Shape: (n_channels,)
        self._rolling_mean = np.zeros(n_channels, dtype=np.float64)
        self._rolling_m2 = np.zeros(n_channels, dtype=np.float64)
        self._rolling_count = 0
        self._sigma_window_samples = int(SIGMA_WINDOW_SEC * sample_rate_hz)

        # EMA-based rolling sigma (for efficiency)
        self._ema_sigma = np.ones(n_channels, dtype=np.float32) * 10.0  # Initial estimate

        # Batch counter
        self._batch_id = 0

    def _design_notch_filters(self) -> list:
        """
        Design IIR notch filters for 60Hz and harmonics.

        Returns:
            List of (b, a) coefficients for each notch frequency.
        """
        coeffs = []
        for freq in NOTCH_FREQUENCIES_HZ:
            # Skip frequencies above Nyquist
            if freq >= self.nyquist:
                continue
            b, a = signal.iirnotch(freq, NOTCH_QUALITY_FACTOR, self.sample_rate)
            coeffs.append((b, a))
        return coeffs

    def _design_lfp_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design 4th-order Butterworth lowpass for LFP extraction.

        Returns:
            (b, a) filter coefficients
        """
        # Normalize cutoff frequency
        wn = LFP_CUTOFF_HZ / self.nyquist
        b, a = signal.butter(BUTTERWORTH_ORDER, wn, btype='low')
        return b, a

    def _design_spike_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design 4th-order Butterworth bandpass for spike extraction.

        Returns:
            (b, a) filter coefficients
        """
        # Normalize cutoff frequencies
        low = SPIKE_LOWCUT_HZ / self.nyquist
        high = SPIKE_HIGHCUT_HZ / self.nyquist
        b, a = signal.butter(BUTTERWORTH_ORDER, [low, high], btype='band')
        return b, a

    def _init_filter_states(self, batch_size: int) -> None:
        """
        Initialize filter states for streaming operation.

        Uses scipy.signal.lfilter_zi for proper initial conditions.
        """
        # Notch filter states
        notch_zi = []
        for b, a in self._notch_coeffs:
            zi = signal.lfilter_zi(b, a)
            # Expand for all channels: shape (filter_order, n_channels)
            zi_expanded = np.tile(zi[:, np.newaxis], (1, self.n_channels))
            notch_zi.append(zi_expanded.copy())

        # LFP filter state
        lfp_zi = signal.lfilter_zi(self._lfp_b, self._lfp_a)
        lfp_zi = np.tile(lfp_zi[:, np.newaxis], (1, self.n_channels))

        # Spike filter state
        spike_zi = signal.lfilter_zi(self._spike_b, self._spike_a)
        spike_zi = np.tile(spike_zi[:, np.newaxis], (1, self.n_channels))

        self._filter_state = FilterState(
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
        batch_size = samples.shape[0]

        # Initialize filter states on first call
        if self._filter_state is None:
            self._init_filter_states(batch_size)

        # Store raw for passthrough
        raw_unfiltered = samples.copy()

        # Step 1: Apply notch filters (cascaded)
        notched = samples.astype(np.float64)  # Higher precision for filtering
        for i, (b, a) in enumerate(self._notch_coeffs):
            notched, self._filter_state.notch_zi[i] = signal.lfilter(
                b, a, notched, axis=0, zi=self._filter_state.notch_zi[i]
            )

        # Step 2: Extract LFP band (lowpass)
        lfp, self._filter_state.lfp_zi = signal.lfilter(
            self._lfp_b, self._lfp_a, notched, axis=0,
            zi=self._filter_state.lfp_zi
        )

        # Step 3: Extract spike band (bandpass)
        spikes, self._filter_state.spike_zi = signal.lfilter(
            self._spike_b, self._spike_a, notched, axis=0,
            zi=self._filter_state.spike_zi
        )

        # Step 4: Update rolling sigma and detect artifacts
        artifact_flags = self._detect_artifacts(notched, batch_size)

        # Increment batch counter
        self._batch_id += 1

        return SanitizedFrame(
            timestamp_us=int(timestamps[0]) if len(timestamps) > 0 else 0,
            raw_unfiltered=raw_unfiltered.astype(np.float32),
            lfp=lfp.astype(np.float32),
            spikes=spikes.astype(np.float32),
            artifact_flags=artifact_flags,
        )

    def _detect_artifacts(
        self,
        data: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        """
        Detect artifacts using Tanh-scaling with rolling sigma.

        Uses EMA for efficient rolling sigma computation.
        Artifacts are flagged per-channel based on batch statistics.

        Args:
            data: Notch-filtered data, shape (batch_size, n_channels)
            batch_size: Number of samples in batch

        Returns:
            Boolean artifact flags, shape (n_channels,)
        """
        # Compute batch statistics per channel
        batch_std = np.std(data, axis=0, dtype=np.float64)
        batch_max = np.max(np.abs(data), axis=0)

        # Update rolling sigma with EMA (Exponential Moving Average)
        # This avoids storing full 10-second history
        if self._rolling_count == 0:
            # First batch: initialize with batch statistics
            self._ema_sigma = batch_std.astype(np.float32)
        else:
            # EMA update: sigma_new = alpha * batch_std + (1-alpha) * sigma_old
            self._ema_sigma = (
                EMA_ALPHA * batch_std + (1 - EMA_ALPHA) * self._ema_sigma
            ).astype(np.float32)

        self._rolling_count += batch_size

        # Flag channels where max amplitude exceeds threshold * rolling sigma
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

        Soft clipping function that compresses large amplitudes.

        Args:
            data: Input data, shape (batch_size, n_channels)
            artifact_flags: Per-channel artifact flags, shape (n_channels,)

        Returns:
            Scaled data with artifacts attenuated
        """
        # Only apply to artifact-flagged channels
        if not np.any(artifact_flags):
            return data

        result = data.copy()

        # For flagged channels, apply tanh scaling
        flagged_channels = np.where(artifact_flags)[0]
        for ch in flagged_channels:
            sigma = self._ema_sigma[ch]
            if sigma > 0:
                # Normalize, apply tanh, denormalize
                threshold = ARTIFACT_THRESHOLD_SIGMA * sigma
                normalized = result[:, ch] / threshold
                result[:, ch] = threshold * np.tanh(normalized)

        return result

    def get_rolling_sigma(self) -> np.ndarray:
        """Get current rolling sigma estimates per channel."""
        return self._ema_sigma.copy()

    def reset(self) -> None:
        """Reset all filter states and rolling statistics."""
        self._filter_state = None
        self._rolling_mean.fill(0)
        self._rolling_m2.fill(0)
        self._rolling_count = 0
        self._ema_sigma.fill(10.0)
        self._batch_id = 0
