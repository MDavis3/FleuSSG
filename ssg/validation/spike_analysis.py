"""Spike buffering and analysis helpers for validation."""

from dataclasses import dataclass

import numpy as np

from ..core.array_types import ChannelIndexVector, FloatMatrix, FloatVector, TimestampVector
from ..core.constants import (
    ISI_REFRACTORY_PERIOD_US,
    MAD_TO_STD_FACTOR,
    MAX_SPIKES_IN_WINDOW,
    SPIKE_DETECTION_THRESHOLD_MAD,
)


@dataclass
class SpikeBuffer:
    """Ring buffer for spike events with vectorized storage."""
    channels: ChannelIndexVector
    timestamps: TimestampVector
    amplitudes: FloatVector
    head: int = 0
    count: int = 0
    capacity: int = MAX_SPIKES_IN_WINDOW

    @classmethod
    def create(cls, capacity: int = MAX_SPIKES_IN_WINDOW) -> "SpikeBuffer":
        """Create an empty spike buffer."""
        return cls(
            channels=np.zeros(capacity, dtype=np.uint16),
            timestamps=np.zeros(capacity, dtype=np.uint64),
            amplitudes=np.zeros(capacity, dtype=np.float32),
            capacity=capacity,
        )

    def add_spikes(
        self,
        channels: ChannelIndexVector,
        timestamps: TimestampVector,
        amplitudes: FloatVector,
    ) -> None:
        """Add multiple spikes without Python loops over entries."""
        n = len(channels)
        if n == 0:
            return

        end = self.head + n
        if end <= self.capacity:
            self.channels[self.head:end] = channels
            self.timestamps[self.head:end] = timestamps
            self.amplitudes[self.head:end] = amplitudes
        else:
            first_part = self.capacity - self.head
            self.channels[self.head:] = channels[:first_part]
            self.timestamps[self.head:] = timestamps[:first_part]
            self.amplitudes[self.head:] = amplitudes[:first_part]

            second_part = n - first_part
            self.channels[:second_part] = channels[first_part:]
            self.timestamps[:second_part] = timestamps[first_part:]
            self.amplitudes[:second_part] = amplitudes[first_part:]

        self.head = end % self.capacity
        self.count = min(self.count + n, self.capacity)

    def get_valid(self) -> tuple[ChannelIndexVector, TimestampVector, FloatVector]:
        """Get all valid spikes in buffer order."""
        if self.count < self.capacity:
            return (
                self.channels[:self.count].copy(),
                self.timestamps[:self.count].copy(),
                self.amplitudes[:self.count].copy(),
            )
        return (
            self.channels.copy(),
            self.timestamps.copy(),
            self.amplitudes.copy(),
        )

    def clear(self) -> None:
        """Reset the visible buffer state."""
        self.head = 0
        self.count = 0

    def trim_before(self, min_timestamp_us: int) -> None:
        """Drop spikes older than the provided timestamp."""
        channels, timestamps, amplitudes = self.get_valid()
        if len(timestamps) == 0:
            return

        keep = timestamps >= min_timestamp_us
        if keep.all():
            return

        self.clear()
        if np.any(keep):
            self.add_spikes(
                channels[keep],
                timestamps[keep],
                amplitudes[keep],
            )


def detect_spikes(
    spike_band: FloatMatrix,
    timestamps: TimestampVector,
    sample_rate: int,
) -> tuple[ChannelIndexVector, TimestampVector, FloatVector]:
    """Detect spikes using local minima and a refractory window."""
    batch_size = spike_band.shape[0]
    if batch_size < 3:
        return (
            np.array([], dtype=np.uint16),
            np.array([], dtype=np.uint64),
            np.array([], dtype=np.float32),
        )

    median = np.median(spike_band, axis=0)
    mad = np.median(np.abs(spike_band - median), axis=0)
    mad = np.maximum(mad, 1e-6)
    threshold = median + SPIKE_DETECTION_THRESHOLD_MAD * mad * MAD_TO_STD_FACTOR

    is_local_min = (
        (spike_band[1:-1] < spike_band[:-2]) &
        (spike_band[1:-1] < spike_band[2:])
    )
    is_spike = is_local_min & (spike_band[1:-1] < threshold)

    time_idx, channel_idx = np.where(is_spike)
    time_idx = time_idx + 1
    if len(time_idx) == 0:
        return (
            np.array([], dtype=np.uint16),
            np.array([], dtype=np.uint64),
            np.array([], dtype=np.float32),
        )

    refractory_samples = int(ISI_REFRACTORY_PERIOD_US * sample_rate / 1_000_000)
    sort_idx = np.lexsort((time_idx, channel_idx))
    time_idx = time_idx[sort_idx]
    channel_idx = channel_idx[sort_idx]

    keep = np.ones(len(time_idx), dtype=bool)
    if len(time_idx) > 1:
        time_diff = np.diff(time_idx)
        chan_diff = np.diff(channel_idx)
        invalid = (chan_diff == 0) & (time_diff < refractory_samples)
        keep[1:] = ~invalid
        time_idx = time_idx[keep]
        channel_idx = channel_idx[keep]

    spike_timestamps = timestamps[time_idx]
    spike_amplitudes = spike_band[time_idx, channel_idx]
    return (
        channel_idx.astype(np.uint16),
        spike_timestamps.astype(np.uint64),
        spike_amplitudes.astype(np.float32),
    )


def compute_isi_violations(
    spike_buffer: SpikeBuffer,
    n_channels: int,
) -> FloatVector:
    """Compute per-channel ISI violation rates."""
    channels, timestamps, _ = spike_buffer.get_valid()
    if len(channels) == 0:
        return np.zeros(n_channels, dtype=np.float32)

    sort_idx = np.lexsort((timestamps, channels))
    sorted_channels = channels[sort_idx]
    sorted_timestamps = timestamps[sort_idx]

    isi = np.diff(sorted_timestamps).astype(np.int64)
    same_channel = np.diff(sorted_channels) == 0
    violations = same_channel & (isi < ISI_REFRACTORY_PERIOD_US)

    violation_channels = sorted_channels[:-1][violations]
    violation_counts = np.bincount(
        violation_channels,
        minlength=n_channels,
    ).astype(np.float32)
    spike_counts = np.bincount(
        channels,
        minlength=n_channels,
    ).astype(np.float32)

    spike_pairs = np.maximum(spike_counts - 1, 1)
    return violation_counts / spike_pairs


def compute_firing_rate(
    spike_buffer: SpikeBuffer,
    n_channels: int,
    observed_duration_sec: float,
) -> FloatVector:
    """Compute per-channel firing rate over the active validation window."""
    if observed_duration_sec <= 0:
        return np.zeros(n_channels, dtype=np.float32)

    channels, _, _ = spike_buffer.get_valid()
    if len(channels) == 0:
        return np.zeros(n_channels, dtype=np.float32)

    spike_counts = np.bincount(channels, minlength=n_channels).astype(np.float32)
    return spike_counts / observed_duration_sec
