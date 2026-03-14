"""Validation engine orchestration."""

import numpy as np

from ..core.array_types import FloatVector, TimestampVector
from ..core.constants import N_CHANNELS, SAMPLE_RATE_HZ, VALIDATION_WINDOW_SEC
from ..core.data_types import ChannelMetrics, RegionMetrics, SanitizedFrame
from .metrics import (
    build_viability_mask,
    summarize_region,
    summarize_spike_band,
    update_ema_snr_from_summary,
)
from .spike_analysis import (
    SpikeBuffer,
    compute_firing_rate,
    compute_isi_violations,
    detect_spikes,
)


class ValidationEngine:
    """Compute per-channel quality metrics for viability scoring."""

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        sample_rate_hz: int = SAMPLE_RATE_HZ,
    ):
        self.n_channels = n_channels
        self.sample_rate = sample_rate_hz
        self._ema_noise = np.ones(n_channels, dtype=np.float32) * 10.0
        self._ema_signal = np.zeros(n_channels, dtype=np.float32)
        self._spike_buffer = SpikeBuffer.create()
        self._impedance_kohm = np.full(n_channels, 1000.0, dtype=np.float32)
        self._batch_count = 0
        self._observed_duration_sec = 0.0

    def process(
        self,
        sanitized: SanitizedFrame,
        timestamps: TimestampVector,
    ) -> ChannelMetrics:
        """Process one sanitized batch and compute quality metrics."""

        batch_duration_sec = sanitized.spikes.shape[0] / self.sample_rate
        spike_band_summary = summarize_spike_band(sanitized.spikes)
        spike_channels, spike_times, spike_amps = detect_spikes(
            sanitized.spikes,
            timestamps,
            self.sample_rate,
            summary=spike_band_summary,
        )
        self._update_spike_history(
            spike_channels,
            spike_times,
            spike_amps,
            timestamps,
            batch_duration_sec,
        )

        snr, self._ema_noise, self._ema_signal = update_ema_snr_from_summary(
            spike_band_summary,
            self._ema_noise,
            self._ema_signal,
            self._batch_count,
        )
        isi_violation_rate = compute_isi_violations(self._spike_buffer, self.n_channels)
        firing_rate_hz = compute_firing_rate(
            self._spike_buffer,
            self.n_channels,
            self._observed_duration_sec,
        )
        viability_mask = build_viability_mask(
            snr,
            isi_violation_rate,
            self._impedance_kohm,
            sanitized.artifact_flags,
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

    def _update_spike_history(
        self,
        spike_channels: FloatVector,
        spike_times: TimestampVector,
        spike_amps: FloatVector,
        timestamps: TimestampVector,
        batch_duration_sec: float,
    ) -> None:
        """Update rolling spike state for ISI and firing-rate metrics."""

        self._spike_buffer.add_spikes(spike_channels, spike_times, spike_amps)
        if len(timestamps) > 0:
            window_start_us = int(timestamps[-1]) - int(VALIDATION_WINDOW_SEC * 1_000_000)
            self._spike_buffer.trim_before(window_start_us)
        self._observed_duration_sec = min(
            self._observed_duration_sec + batch_duration_sec,
            VALIDATION_WINDOW_SEC,
        )

    def update_impedance(self, impedance_kohm: FloatVector) -> None:
        """Update electrode impedance values."""

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
        """Get aggregated metrics for a channel region."""

        return summarize_region(metrics, start_ch, end_ch)

    def reset(self) -> None:
        """Reset all statistics."""

        self._ema_noise.fill(10.0)
        self._ema_signal.fill(0.0)
        self._spike_buffer.clear()
        self._observed_duration_sec = 0.0
        self._batch_count = 0
