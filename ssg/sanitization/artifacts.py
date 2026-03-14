"""Artifact-state helpers for sanitization."""

import numpy as np

from ..core.array_types import BoolVector, FloatMatrix, FloatVector
from ..core.constants import ARTIFACT_THRESHOLD_SIGMA, EMA_ALPHA


def update_artifact_state(
    data: FloatMatrix,
    ema_sigma: FloatVector,
    rolling_count: int,
) -> tuple[BoolVector, FloatVector, int]:
    """Update the rolling sigma estimate and return artifact flags."""

    batch_std = np.std(data, axis=0, dtype=np.float64)
    batch_max = np.max(np.abs(data), axis=0)

    if rolling_count == 0:
        next_sigma = batch_std.astype(np.float32)
    else:
        next_sigma = (
            EMA_ALPHA * batch_std + (1 - EMA_ALPHA) * ema_sigma
        ).astype(np.float32)

    threshold = ARTIFACT_THRESHOLD_SIGMA * next_sigma
    artifact_flags = batch_max > threshold
    return artifact_flags.astype(bool), next_sigma, rolling_count + data.shape[0]


def apply_tanh_scaling(
    data: FloatMatrix,
    artifact_flags: BoolVector,
    ema_sigma: FloatVector,
) -> FloatMatrix:
    """Attenuate artifact-heavy channels while preserving waveform shape."""

    if not np.any(artifact_flags):
        return data

    result = data.copy()
    threshold = np.maximum(ARTIFACT_THRESHOLD_SIGMA * ema_sigma, 1e-6)
    normalized = result[:, artifact_flags] / threshold[artifact_flags]
    result[:, artifact_flags] = threshold[artifact_flags] * np.tanh(normalized)
    return result
