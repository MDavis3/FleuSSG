"""Streaming sanitization pipeline."""

from typing import Optional

import numpy as np

from ..core.array_types import BoolVector, FloatMatrix, FloatVector, TimestampVector
from ..core.constants import N_CHANNELS, SAMPLE_RATE_HZ
from ..core.data_types import SanitizedFrame
from .artifacts import apply_tanh_scaling, update_artifact_state
from .filters import SOSFilterState, apply_filter_bank, design_filter_bank, init_filter_state


class SanitizationLayer:
    """Run the streaming DSP pipeline for neural signals."""

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        sample_rate_hz: int = SAMPLE_RATE_HZ,
    ):
        self.n_channels = n_channels
        self.sample_rate = sample_rate_hz
        self._filter_bank = design_filter_bank(sample_rate_hz)
        self._filter_state: Optional[SOSFilterState] = None
        self._ema_sigma = np.ones(n_channels, dtype=np.float32) * 50.0
        self._rolling_count = 0
        self._batch_id = 0

    def _ensure_filter_state(self) -> SOSFilterState:
        """Initialize streaming state on the first batch."""

        if self._filter_state is None:
            self._filter_state = init_filter_state(self._filter_bank, self.n_channels)
        return self._filter_state

    def sanitize_batch(
        self,
        samples: FloatMatrix,
        timestamps: TimestampVector,
    ) -> SanitizedFrame:
        """Sanitize one sample batch through the streaming DSP pipeline."""

        filter_state = self._ensure_filter_state()
        raw_unfiltered = samples.copy()
        notched, lfp, spikes = apply_filter_bank(samples, self._filter_bank, filter_state)
        artifact_flags, self._ema_sigma, self._rolling_count = update_artifact_state(
            notched,
            self._ema_sigma,
            self._rolling_count,
        )
        self._batch_id += 1

        return SanitizedFrame(
            timestamp_us=int(timestamps[0]) if len(timestamps) > 0 else 0,
            raw_unfiltered=raw_unfiltered.astype(np.float32),
            lfp=lfp.astype(np.float32),
            spikes=spikes.astype(np.float32),
            artifact_flags=artifact_flags,
        )

    def apply_tanh_scaling(
        self,
        data: FloatMatrix,
        artifact_flags: BoolVector,
    ) -> FloatMatrix:
        """Apply Tanh scaling to channels flagged as artifactual."""

        return apply_tanh_scaling(data, artifact_flags, self._ema_sigma)

    def get_rolling_sigma(self) -> FloatVector:
        """Get current rolling sigma estimates per channel."""

        return self._ema_sigma.copy()

    def reset(self) -> None:
        """Reset all filter states and rolling statistics."""

        self._filter_state = None
        self._rolling_count = 0
        self._ema_sigma.fill(10.0)
        self._batch_id = 0