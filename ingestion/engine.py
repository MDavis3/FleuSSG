"""
IngestionEngine for Signal Stability Gateway

Manages continuous 1024-channel data acquisition into a ring buffer.
Designed for real-time streaming with zero-copy operations.
"""

import numpy as np
from typing import Tuple, Optional, Callable
import time

from ..core.ring_buffer import RingBuffer
from ..core.constants import (
    N_CHANNELS,
    SAMPLE_RATE_HZ,
    BATCH_SIZE,
)


class IngestionEngine:
    """
    Manages continuous 1024-channel data acquisition into a ring buffer.

    Data Flow:
        [Neural Array / Mock Telemetry] → [Ring Buffer] → [Sanitization Layer]

    FDA Documentation:
        - Sampling: 20 kHz per channel
        - Resolution: 16-bit ADC (±3.3V range)
        - Buffer: 2-second capacity (40,000 samples)
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        sample_rate_hz: int = SAMPLE_RATE_HZ,
        buffer_duration_sec: float = 2.0,
        data_source: Optional[Callable[[], Tuple[np.ndarray, int]]] = None,
    ):
        """
        Initialize IngestionEngine.

        Args:
            n_channels: Number of recording channels (default: 1024)
            sample_rate_hz: Sampling rate in Hz (default: 20000)
            buffer_duration_sec: Ring buffer capacity in seconds
            data_source: Optional callback that returns (samples, timestamp_us)
        """
        self.n_channels = n_channels
        self.sample_rate = sample_rate_hz
        self.buffer_duration = buffer_duration_sec
        self.buffer_size = int(sample_rate_hz * buffer_duration_sec)

        # Pre-allocated ring buffer (zero-copy design)
        self._buffer = RingBuffer(
            shape=(self.buffer_size, n_channels),
            dtype=np.float32
        )

        # Data source callback (for real hardware or mock)
        self._data_source = data_source

        # Statistics
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

    def ingest_frame(self, samples: np.ndarray, timestamp_us: int) -> None:
        """
        Push a single frame (1024 samples) into the ring buffer.

        Vectorized - no Python loops.

        Args:
            samples: shape (n_channels,) - one sample per channel
            timestamp_us: Microsecond timestamp
        """
        assert samples.shape == (self.n_channels,), \
            f"Expected shape ({self.n_channels},), got {samples.shape}"

        self._buffer.push(samples, timestamp_us)
        self._total_frames_ingested += 1
        self._last_timestamp_us = timestamp_us

    def ingest_batch(self, samples: np.ndarray, timestamps: np.ndarray) -> None:
        """
        Push a batch of frames into the ring buffer.

        VECTORIZED - uses block assignment, no Python loops.

        Args:
            samples: shape (N, n_channels) - batch of samples
            timestamps: shape (N,) - microsecond timestamps
        """
        assert samples.shape[1] == self.n_channels, \
            f"Expected {self.n_channels} channels, got {samples.shape[1]}"
        assert samples.shape[0] == timestamps.shape[0], \
            "Samples and timestamps must have same length"

        self._buffer.push_batch(samples, timestamps)
        self._total_frames_ingested += samples.shape[0]
        self._last_timestamp_us = timestamps[-1]

    def get_batch(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve last N samples for batch processing.

        Args:
            n_samples: Number of samples to retrieve

        Returns:
            Tuple of (samples [N, 1024], timestamps [N])
        """
        return self._buffer.get_last(n_samples)

    def get_latest_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the default batch size (BATCH_SIZE from constants).

        Returns:
            Tuple of (samples [BATCH_SIZE, 1024], timestamps [BATCH_SIZE])
        """
        return self._buffer.get_last(BATCH_SIZE)

    def poll(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Poll the data source for new data (if configured).

        Returns:
            Tuple of (samples, timestamps) if data available, None otherwise
        """
        if self._data_source is None:
            return None

        samples, timestamp = self._data_source()
        if samples is not None:
            if samples.ndim == 1:
                self.ingest_frame(samples, timestamp)
                return samples.reshape(1, -1), np.array([timestamp])
            else:
                # Generate timestamps for batch
                n_samples = samples.shape[0]
                sample_interval_us = int(1_000_000 / self.sample_rate)
                timestamps = timestamp - np.arange(n_samples - 1, -1, -1) * sample_interval_us
                self.ingest_batch(samples, timestamps)
                return samples, timestamps

        return None

    def clear(self) -> None:
        """Clear the buffer and reset statistics."""
        self._buffer.clear()
        self._total_frames_ingested = 0
        self._last_timestamp_us = 0

    def get_buffer_fill_ratio(self) -> float:
        """Get current buffer fill ratio (0.0 to 1.0)."""
        return self._buffer.current_size / self.buffer_size
