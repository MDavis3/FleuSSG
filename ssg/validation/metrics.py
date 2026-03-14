"""Pure metric helpers for validation."""

import numpy as np

from ..core.array_types import BoolVector, FloatMatrix, FloatVector
from ..core.constants import (
    EMA_ALPHA,
    IMPEDANCE_MAX_KOHM,
    IMPEDANCE_MIN_KOHM,
    ISI_VIOLATION_LIMIT,
    MAD_TO_STD_FACTOR,
    SNR_THRESHOLD,
)
from ..core.data_types import ChannelMetrics, RegionMetrics


def update_ema_snr(
    spike_band: FloatMatrix,
    ema_noise: FloatVector,
    ema_signal: FloatVector,
    batch_count: int,
) -> tuple[FloatVector, FloatVector, FloatVector]:
    """Update EMA-backed noise and signal estimates and return SNR."""

    median = np.median(spike_band, axis=0)
    mad = np.median(np.abs(spike_band - median), axis=0)
    batch_noise = np.maximum(mad * MAD_TO_STD_FACTOR, 1e-6)
    batch_signal = np.percentile(np.abs(spike_band), 95, axis=0)

    if batch_count == 0:
        next_noise = batch_noise.astype(np.float32)
        next_signal = batch_signal.astype(np.float32)
    else:
        next_noise = (
            EMA_ALPHA * batch_noise + (1 - EMA_ALPHA) * ema_noise
        ).astype(np.float32)
        next_signal = (
            EMA_ALPHA * batch_signal + (1 - EMA_ALPHA) * ema_signal
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
