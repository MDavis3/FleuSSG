"""Helpers for normalizing polled ingestion batches."""

import numpy as np

from ..core.array_types import FloatMatrix, TimestampVector


def build_batch_timestamps(
    sample_count: int,
    sample_rate_hz: int,
    latest_timestamp_us: int,
) -> TimestampVector:
    """Generate evenly spaced timestamps ending at the latest sample."""

    sample_interval_us = int(1_000_000 / sample_rate_hz)
    offsets = np.arange(sample_count - 1, -1, -1, dtype=np.int64) * sample_interval_us
    return (latest_timestamp_us - offsets).astype(np.uint64)


def normalize_polled_batch(
    samples: np.ndarray,
    timestamp_us: int,
    sample_rate_hz: int,
) -> tuple[FloatMatrix, TimestampVector]:
    """Normalize a polled frame or batch into batch-shaped arrays."""

    if samples.ndim == 1:
        return samples.reshape(1, -1), np.array([timestamp_us], dtype=np.uint64)

    return samples, build_batch_timestamps(samples.shape[0], sample_rate_hz, timestamp_us)
