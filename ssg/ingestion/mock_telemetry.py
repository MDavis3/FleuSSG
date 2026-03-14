"""
Mock Telemetry Generator for SSG Testing

Generates synthetic 1024-channel neural data for pipeline validation.
Includes realistic spike patterns and configurable noise levels.
"""

from dataclasses import dataclass
import time
from typing import Optional, Tuple

import numpy as np

from ..core.constants import (
    N_CHANNELS,
    SAMPLE_RATE_HZ,
    BATCH_SIZE,
    SYNTHETIC_SPIKE_RATE_HZ,
    SPIKE_TEMPLATE_DURATION_SEC,
    SYNTHETIC_NOISE_AMPLITUDE_UV,
    SYNTHETIC_SPIKE_AMPLITUDE_UV,
)


@dataclass(frozen=True)
class MockTelemetryConfig:
    """Tunable parameters for synthetic telemetry generation."""
    noise_amplitude_uv: float = SYNTHETIC_NOISE_AMPLITUDE_UV
    spike_amplitude_uv: float = SYNTHETIC_SPIKE_AMPLITUDE_UV
    spike_rate_hz: float = SYNTHETIC_SPIKE_RATE_HZ
    seed: Optional[int] = None


class MockTelemetry:
    """
    Generates synthetic 1024-channel neural telemetry.

    Features:
        - Gaussian background noise (configurable amplitude)
        - Realistic biphasic spike templates
        - Per-channel spike rate variation
        - Real-time timestamp generation

    Usage:
        mock = MockTelemetry()
        samples, timestamp = mock.generate_batch(batch_size=2000)
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        sample_rate_hz: int = SAMPLE_RATE_HZ,
        config: Optional[MockTelemetryConfig] = None,
    ):
        """
        Initialize mock telemetry generator.

        Args:
            n_channels: Number of channels to simulate
            sample_rate_hz: Sampling rate
            config: Synthetic telemetry configuration
        """
        telemetry_config = config or MockTelemetryConfig()

        self.n_channels = n_channels
        self.sample_rate = sample_rate_hz
        self.noise_amplitude = telemetry_config.noise_amplitude_uv
        self.spike_amplitude = telemetry_config.spike_amplitude_uv
        self.spike_rate = telemetry_config.spike_rate_hz

        # Random number generator
        self.rng = np.random.default_rng(telemetry_config.seed)

        # Pre-compute spike template (biphasic extracellular spike)
        self._spike_template = self._create_spike_template()
        self._scaled_spike_template = self._spike_template * np.float32(
            self.spike_amplitude
        )

        # Per-channel spike rates (some variation)
        self._channel_spike_rates = self.rng.uniform(
            0.5 * self.spike_rate,
            1.5 * self.spike_rate,
            size=n_channels
        ).astype(np.float32)

        # Timestamp tracking
        self._start_time_us: Optional[int] = None
        self._sample_counter = 0

    def _create_spike_template(self) -> np.ndarray:
        """
        Create a realistic biphasic extracellular spike template.

        Shape: Initial negative deflection followed by positive overshoot.
        Duration: ~2ms (SPIKE_TEMPLATE_DURATION_SEC)
        """
        n_samples = int(SPIKE_TEMPLATE_DURATION_SEC * self.sample_rate)
        t = np.arange(n_samples)

        # Biphasic waveform: negative peak followed by positive recovery
        # Models typical extracellular action potential
        tau = n_samples / 4
        template = (
            -np.exp(-t / tau) * np.sin(2 * np.pi * t / n_samples)
            + 0.3 * np.exp(-(t - n_samples/2)**2 / (2 * (tau/2)**2))
        )

        # Normalize to unit amplitude
        template = template / np.abs(template).max()

        return template.astype(np.float32)

    def generate_batch(
        self,
        batch_size: int = BATCH_SIZE,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a batch of synthetic neural data.

        Args:
            batch_size: Number of samples to generate

        Returns:
            Tuple of (samples [batch_size, n_channels], timestamps [batch_size])
        """
        # Generate background noise
        samples = self.rng.standard_normal(
            (batch_size, self.n_channels),
            dtype=np.float32,
        ) * np.float32(self.noise_amplitude)

        # Add spikes to each channel
        samples = self._add_spikes(samples, batch_size)

        # Generate timestamps
        sample_interval_us = int(1_000_000 / self.sample_rate)
        if self._start_time_us is None:
            self._start_time_us = int(time.time() * 1_000_000)
        start_ts = self._start_time_us + self._sample_counter * sample_interval_us
        timestamps = start_ts + np.arange(batch_size) * sample_interval_us

        # Update counter
        self._sample_counter += batch_size

        return samples, timestamps.astype(np.uint64)

    def _add_spikes(
        self,
        samples: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        """
        Add realistic spikes to the signal.

        Uses Poisson process for spike timing.
        """
        template_len = len(self._scaled_spike_template)
        duration_sec = batch_size / self.sample_rate
        max_start = batch_size - template_len + 1

        if max_start <= 0:
            return samples

        spike_counts = self.rng.poisson(
            self._channel_spike_rates * duration_sec
        ).astype(np.int32, copy=False)
        active_channels = np.flatnonzero(spike_counts)
        if len(active_channels) == 0:
            return samples

        active_counts = spike_counts[active_channels]
        total_spikes = int(active_counts.sum())
        spike_times = self.rng.integers(
            0,
            max_start,
            size=total_spikes,
            dtype=np.int32,
        )
        spike_channels = np.repeat(
            active_channels.astype(np.intp, copy=False),
            active_counts,
        )
        amplitude_factors = self.rng.uniform(
            0.7,
            1.3,
            size=total_spikes,
        ).astype(np.float32)

        spike_starts = np.zeros_like(samples)
        np.add.at(spike_starts, (spike_times, spike_channels), amplitude_factors)

        for offset, weight in enumerate(self._scaled_spike_template):
            end = batch_size - offset
            samples[offset:offset + end] += spike_starts[:end] * weight

        return samples

    def generate_frame(self) -> Tuple[np.ndarray, int]:
        """
        Generate a single frame (for real-time simulation).

        Returns:
            Tuple of (samples [n_channels], timestamp_us)
        """
        samples, timestamps = self.generate_batch(batch_size=1)
        return samples[0], timestamps[0]

    def reset(self) -> None:
        """Reset timestamp counter."""
        self._start_time_us = None
        self._sample_counter = 0

    def get_elapsed_time_sec(self) -> float:
        """Get elapsed simulation time in seconds."""
        return self._sample_counter / self.sample_rate


class RealtimeMockTelemetry(MockTelemetry):
    """
    Real-time mock telemetry with timing control.

    Generates data at the actual sampling rate for live testing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_generate_time = None

    def generate_realtime_batch(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate batch only if enough real time has passed.

        Returns:
            Data if timing interval reached, None otherwise
        """
        current_time = time.time()

        if self._last_generate_time is None:
            self._last_generate_time = current_time
            return self.generate_batch()

        elapsed = current_time - self._last_generate_time
        expected_samples = int(elapsed * self.sample_rate)

        if expected_samples >= BATCH_SIZE:
            self._last_generate_time = current_time
            return self.generate_batch(batch_size=min(expected_samples, BATCH_SIZE * 2))

        return None
