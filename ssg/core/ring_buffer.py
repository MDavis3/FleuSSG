"""
Lock-Free Ring Buffer for Real-Time Neural Data

Optimized for zero-copy operations with NumPy arrays.
No memory allocation after initialization.
"""

import numpy as np
from typing import Tuple


class RingBuffer:
    """
    Lock-free ring buffer for real-time 1024-channel neural data.

    Memory Layout:
    - Contiguous NumPy array with wraparound indexing
    - Pre-allocated at initialization (no runtime allocation)
    - Supports vectorized batch insertion and retrieval

    Usage:
        buffer = RingBuffer(shape=(40000, 1024), dtype=np.float32)
        buffer.push_batch(samples, timestamps)
        recent_data, recent_ts = buffer.get_last(1000)
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        dtype: np.dtype = np.float32,
    ):
        """
        Initialize ring buffer.

        Args:
            shape: (buffer_size, n_channels) - e.g., (40000, 1024) for 2s at 20kHz
            dtype: NumPy dtype for samples
        """
        self._size = shape[0]
        self._n_channels = shape[1]
        self._dtype = dtype

        # Pre-allocate data storage
        self._data = np.zeros(shape, dtype=dtype)
        self._timestamps = np.zeros(self._size, dtype=np.uint64)

        # Write position (head)
        self._head = 0

        # Track if buffer has wrapped (is full)
        self._full = False

        # Total samples written (for debugging)
        self._total_written = 0

    @property
    def is_full(self) -> bool:
        """Return True if buffer has been completely filled at least once."""
        return self._full

    @property
    def current_size(self) -> int:
        """Return number of valid samples in buffer."""
        if self._full:
            return self._size
        return self._head

    def push(self, sample: np.ndarray, timestamp: int) -> None:
        """
        Push a single sample (all channels) to the buffer.

        O(1) insertion with no memory allocation.

        Args:
            sample: shape (n_channels,) - single time point, all channels
            timestamp: Microsecond timestamp
        """
        self._data[self._head] = sample
        self._timestamps[self._head] = timestamp

        self._head = (self._head + 1) % self._size
        self._total_written += 1

        if self._head == 0:
            self._full = True

    def push_batch(self, samples: np.ndarray, timestamps: np.ndarray) -> None:
        """
        VECTORIZED batch insertion using block assignment.

        No Python loops. Handles wraparound with two slices.

        Args:
            samples: shape (N, n_channels) - batch of samples
            timestamps: shape (N,) - timestamps for each sample
        """
        n_samples = samples.shape[0]

        if n_samples == 0:
            return

        # Calculate write range
        start = self._head
        end = start + n_samples

        if end <= self._size:
            # Simple case: no wraparound
            self._data[start:end] = samples
            self._timestamps[start:end] = timestamps
        else:
            # Wraparound case: two slices
            first_part = self._size - start

            # Write first part (end of buffer)
            self._data[start:] = samples[:first_part]
            self._timestamps[start:] = timestamps[:first_part]

            # Write second part (beginning of buffer)
            second_part = n_samples - first_part
            self._data[:second_part] = samples[first_part:]
            self._timestamps[:second_part] = timestamps[first_part:]

            self._full = True

        # Update head position
        self._head = end % self._size
        self._total_written += n_samples

        if end >= self._size:
            self._full = True

    def get_last(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the last N samples from the buffer.

        Returns contiguous copy (required for downstream processing).

        Args:
            n: Number of samples to retrieve

        Returns:
            Tuple of (samples [N, n_channels], timestamps [N])
        """
        available = self.current_size
        n = min(n, available)

        if n == 0:
            return (
                np.empty((0, self._n_channels), dtype=self._dtype),
                np.empty(0, dtype=np.uint64)
            )

        # Calculate read range (going backwards from head)
        end = self._head
        start = end - n

        if start >= 0:
            # Simple case: no wraparound
            return (
                self._data[start:end].copy(),
                self._timestamps[start:end].copy()
            )
        else:
            # Wraparound case: concatenate two slices
            first_start = self._size + start  # Negative wraps to end
            first_part = self._data[first_start:]
            second_part = self._data[:end]

            first_ts = self._timestamps[first_start:]
            second_ts = self._timestamps[:end]

            return (
                np.concatenate([first_part, second_part], axis=0),
                np.concatenate([first_ts, second_ts], axis=0)
            )

    def get_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all valid samples in chronological order.

        Returns:
            Tuple of (samples, timestamps) in order from oldest to newest
        """
        if not self._full:
            return (
                self._data[:self._head].copy(),
                self._timestamps[:self._head].copy()
            )

        # Buffer is full, reorder from head (oldest) to head-1 (newest)
        return (
            np.concatenate([self._data[self._head:], self._data[:self._head]], axis=0),
            np.concatenate([self._timestamps[self._head:], self._timestamps[:self._head]], axis=0)
        )

    def clear(self) -> None:
        """Reset buffer to empty state (no deallocation)."""
        self._head = 0
        self._full = False
        self._total_written = 0
        # Note: We don't zero the arrays for performance
