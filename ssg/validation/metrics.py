"""Pure metric helpers for validation."""

from dataclasses import dataclass

import numpy as np

from ..core.array_types import BoolVector, FloatMatrix, FloatVector
from ..core.constants import (
    EMA_ALPHA,
    IMPEDANCE_MAX_KOHM,
    IMPEDANCE_MIN_KOHM,
    ISI_VIOLATION_LIMIT,
    MAD_TO_STD_FACTOR,
    SNR_THRESHOLD,
    SPIKE_DETECTION_THRESHOLD_MAD,
    VALIDATION_SUMMARY_STRIDE,
)
from ..core.data_types import ChannelMetrics, RegionMetrics


@dataclass(frozen=True)
class SpikeBandSummary:
    """Robust per-channel summary statistics for one spike-band batch."""

    median: FloatVector
    mad: FloatVector
    detection_threshold: FloatVector
    batch_noise: FloatVector
    batch_signal: FloatVector


def summarize_spike_band(spike_band: FloatMatrix) -> SpikeBandSummary:
    """Summarize the spike band once so downstream stages can reuse it."""

    summary_source = _select_summary_source(spike_band)
    median = np.median(summary_source, axis=0).astype(np.float32, copy=False)
    absolute_deviation = np.abs(summary_source - median)
    mad = np.maximum(
        np.median(absolute_deviation, axis=0),
        1e-6,
    ).astype(np.float32, copy=False)
    batch_noise = np.maximum(
        mad * MAD_TO_STD_FACTOR,
        1e-6,
    ).astype(np.float32, copy=False)

    signal_rank = max(int(np.floor(0.95 * (summary_source.shape[0] - 1))), 0)
    absolute_band = np.abs(summary_source)
    batch_signal = np.partition(absolute_band, signal_rank, axis=0)[
        signal_rank
    ].astype(np.float32, copy=False)
    detection_threshold = (
        median + SPIKE_DETECTION_THRESHOLD_MAD * batch_noise
    ).astype(np.float32, copy=False)
    return SpikeBandSummary(
        median=median,
        mad=mad,
        detection_threshold=detection_threshold,
        batch_noise=batch_noise,
        batch_signal=batch_signal,
    )


def _select_summary_source(spike_band: FloatMatrix) -> FloatMatrix:
    """Subsample large batches to reduce robust-statistic cost."""

    if spike_band.shape[0] < VALIDATION_SUMMARY_STRIDE * 16:
        return spike_band
    return spike_band[::VALIDATION_SUMMARY_STRIDE]


def update_ema_snr(
    spike_band: FloatMatrix,
    ema_noise: FloatVector,
    ema_signal: FloatVector,
    batch_count: int,
) -> tuple[FloatVector, FloatVector, FloatVector]:
    """Update EMA-backed noise and signal estimates and return SNR."""

    return update_ema_snr_from_summary(
        summarize_spike_band(spike_band),
        ema_noise,
        ema_signal,
        batch_count,
    )


def update_ema_snr_from_summary(
    summary: SpikeBandSummary,
    ema_noise: FloatVector,
    ema_signal: FloatVector,
    batch_count: int,
) -> tuple[FloatVector, FloatVector, FloatVector]:
    """Update EMA-backed noise and signal estimates from precomputed stats."""

    if batch_count == 0:
        next_noise = summary.batch_noise.astype(np.float32, copy=False)
        next_signal = summary.batch_signal.astype(np.float32, copy=False)
    else:
        next_noise = (
            EMA_ALPHA * summary.batch_noise + (1 - EMA_ALPHA) * ema_noise
        ).astype(np.float32)
        next_signal = (
            EMA_ALPHA * summary.batch_signal + (1 - EMA_ALPHA) * ema_signal
        ).astype(np.float32)

    snr = next_signal / next_noise
    return snr.astype(np.float32), next_noise, next_signal


def build_viability_mask(
    snr: FloatVector,
    isi_violation_rate: FloatVector,
    impedance_kohm: FloatVector,
    artifact_flags: BoolVector,
) -> BoolVector:
    """Apply the viability criteria for a batch of channels."""

    return (
        (snr >= SNR_THRESHOLD)
        & (isi_violation_rate < ISI_VIOLATION_LIMIT)
        & (impedance_kohm >= IMPEDANCE_MIN_KOHM)
        & (impedance_kohm <= IMPEDANCE_MAX_KOHM)
        & (~artifact_flags)
    ).astype(bool)


def summarize_region(
    metrics: ChannelMetrics,
    start_ch: int,
    end_ch: int,
) -> RegionMetrics:
    """Aggregate channel metrics for a contiguous region."""

    region_mask = metrics.viability_mask[start_ch:end_ch]
    region_snr = metrics.snr[start_ch:end_ch]
    region_fr = metrics.firing_rate_hz[start_ch:end_ch]
    total_count = end_ch - start_ch

    return RegionMetrics(
        viable_count=int(region_mask.sum()),
        total_count=total_count,
        viability_pct=100.0 * float(region_mask.sum()) / total_count,
        mean_snr=float(np.mean(region_snr)),
        mean_firing_rate_hz=float(np.mean(region_fr)),
    )
