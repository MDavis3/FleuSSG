"""Streaming ingestion engine."""

from typing import Callable, Optional, Tuple

import numpy as np

from ..core.array_types import FloatMatrix, FloatVector, TimestampVector
from ..core.constants import BATCH_SIZE, N_CHANNELS, SAMPLE_RATE_HZ
from ..core.ring_buffer import RingBuffer
from .polling import normalize_polled_batch


class IngestionEngine:
    """Manage continuous data acquisition into the ring buffer."""

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        sample_rate_hz: int = SAMPLE_RATE_HZ,
        buffer_duration_sec: float = 2.0,
        data_source: Optional[Callable[[], Tuple[np.ndarray, int]]] = None,
    ):
        self.n_channels = n_channels
        self.sample_rate = sample_rate_hz
        self.buffer_duration = buffer_duration_sec
        self.buffer_size = int(sample_rate_hz * buffer_duration_sec)
        self._buffer = RingBuffer(shape=(self.buffer_size, n_channels), dtype=np.float32)
        self._data_source = data_source
        self._total_frames_ingested = 0
        self._last_timestamp_us = 0

    @property
    def buffer(self) -> RingBuffer:
        """Access underlying ring buffer."""

        return self._buffer

    @property
    def total_frames(self) -> int:
        """Total frames ingested since initialization."""

        return self._total_frames_ingested

    def ingest_frame(self, samples: FloatVector, timestamp_us: int) -> None:
        """Push one frame into the ring buffer."""

        if samples.shape != (self.n_channels,):
            raise ValueError(f"Expected shape ({self.n_channels},), got {samples.shape}")

        self._buffer.push(samples, timestamp_us)
        self._total_frames_ingested += 1
        self._last_timestamp_us = timestamp_us

    def ingest_batch(self, samples: FloatMatrix, timestamps: TimestampVector) -> None:
        """Push a batch of frames into the ring buffer."""

        if samples.ndim != 2:
            raise ValueError(f"samples must be 2D with shape (batch, {self.n_channels})")
        if samples.shape[1] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {samples.shape[1]}")
        if samples.shape[0] != timestamps.shape[0]:
            raise ValueError("Samples and timestamps must have same length")

        self._buffer.push_batch(samples, timestamps)
        self._total_frames_ingested += samples.shape[0]
        self._last_timestamp_us = int(timestamps[-1])

    def get_batch(self, n_samples: int) -> Tuple[FloatMatrix, TimestampVector]:
        """Retrieve the last ``n_samples`` frames."""

        return self._buffer.get_last(n_samples)

    def get_latest_batch(self) -> Tuple[FloatMatrix, TimestampVector]:
        """Get the default batch size."""

        return self._buffer.get_last(BATCH_SIZE)

    def poll(self) -> Optional[Tuple[FloatMatrix, TimestampVector]]:
        """Poll the data source and ingest any returned batch."""

        if self._data_source is None:
            return None

        samples, timestamp = self._data_source()
        if samples is None:
            return None

        batch_samples, timestamps = normalize_polled_batch(
            samples,
            timestamp,
            self.sample_rate,
        )
        if samples.ndim == 1:
            self.ingest_frame(samples, timestamp)
        else:
            self.ingest_batch(batch_samples, timestamps)
        return batch_samples, timestamps

    def clear(self) -> None:
        """Clear the buffer and reset statistics."""

        self._buffer.clear()
        self._total_frames_ingested = 0
        self._last_timestamp_us = 0

    def get_buffer_fill_ratio(self) -> float:
        """Get current buffer fill ratio (0.0 to 1.0)."""

        return self._buffer.current_size / self.buffer_size
