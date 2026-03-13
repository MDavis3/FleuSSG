"""
Biological Noise Models for SSG Testing

Simulates realistic artifacts for validation testing:
- Jaw Clench: High-amplitude EMG burst
- Electrode Drift: Slow DC shift
- Motion Spike: Sharp transient

All models are parameterized based on clinical observations.
"""

import numpy as np
from enum import Enum, auto
from typing import Tuple, Optional
from dataclasses import dataclass

from ..core.constants import (
    N_CHANNELS,
    SAMPLE_RATE_HZ,
    JAW_CLENCH_PROBABILITY,
    ELECTRODE_DRIFT_PROBABILITY,
    MOTION_SPIKE_PROBABILITY,
)


class ArtifactType(Enum):
    """Types of biological artifacts."""
    JAW_CLENCH = auto()
    ELECTRODE_DRIFT = auto()
    MOTION_SPIKE = auto()


@dataclass
class ArtifactEvent:
    """Record of an injected artifact."""
    artifact_type: ArtifactType
    start_sample: int
    duration_samples: int
    affected_channels: np.ndarray
    amplitude: float


class NoiseGenerator:
    """
    Generates biological noise artifacts for testing.

    Each artifact type models a specific clinical phenomenon:

    Jaw Clench:
        - High-amplitude EMG burst (50-500ms duration)
        - Affects ~30% of channels (muscle group overlap)
        - Amplitude: 5-20x baseline noise

    Electrode Drift:
        - Slow DC shift over ~2 seconds
        - Affects ~5% of channels (electrode contact issues)
        - Amplitude: 3-10x baseline

    Motion Spike:
        - Sharp transient (~10ms)
        - Affects ~80% of channels (mechanical coupling)
        - Amplitude: 10-50x baseline
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        sample_rate_hz: int = SAMPLE_RATE_HZ,
        seed: Optional[int] = None,
    ):
        """
        Initialize noise generator.

        Args:
            n_channels: Number of recording channels
            sample_rate_hz: Sampling rate in Hz
            seed: Random seed for reproducibility
        """
        self.n_channels = n_channels
        self.sample_rate = sample_rate_hz
        self.rng = np.random.default_rng(seed)

        # Track injected artifacts
        self._artifact_history: list = []

    def inject_artifacts(
        self,
        samples: np.ndarray,
        jaw_clench_prob: float = JAW_CLENCH_PROBABILITY,
        electrode_drift_prob: float = ELECTRODE_DRIFT_PROBABILITY,
        motion_spike_prob: float = MOTION_SPIKE_PROBABILITY,
        baseline_noise: float = 10.0,
    ) -> Tuple[np.ndarray, list]:
        """
        Inject biological artifacts into signal.

        Artifacts are applied in-place for efficiency.

        Args:
            samples: Input signal, shape (batch_size, n_channels)
            jaw_clench_prob: Probability of jaw clench per second
            electrode_drift_prob: Probability of electrode drift per second
            motion_spike_prob: Probability of motion spike per second
            baseline_noise: Expected baseline noise amplitude (µV)

        Returns:
            Tuple of (modified samples, list of ArtifactEvent)
        """
        batch_size = samples.shape[0]
        batch_duration_sec = batch_size / self.sample_rate
        injected_artifacts = []

        # Calculate expected number of each artifact type
        n_jaw_clench = self.rng.poisson(jaw_clench_prob * batch_duration_sec)
        n_electrode_drift = self.rng.poisson(electrode_drift_prob * batch_duration_sec)
        n_motion_spike = self.rng.poisson(motion_spike_prob * batch_duration_sec)

        # Inject each artifact type
        for _ in range(n_jaw_clench):
            artifact = self._inject_jaw_clench(samples, baseline_noise)
            if artifact:
                injected_artifacts.append(artifact)

        for _ in range(n_electrode_drift):
            artifact = self._inject_electrode_drift(samples, baseline_noise)
            if artifact:
                injected_artifacts.append(artifact)

        for _ in range(n_motion_spike):
            artifact = self._inject_motion_spike(samples, baseline_noise)
            if artifact:
                injected_artifacts.append(artifact)

        self._artifact_history.extend(injected_artifacts)

        return samples, injected_artifacts

    def _inject_jaw_clench(
        self,
        samples: np.ndarray,
        baseline_noise: float,
    ) -> Optional[ArtifactEvent]:
        """
        Inject jaw clench artifact (EMG burst).

        Characteristics:
            - Duration: 50-500ms
            - Channels: ~30% (random subset)
            - Amplitude: 5-20x baseline
            - Waveform: Band-limited noise (100-500Hz)
        """
        batch_size = samples.shape[0]

        # Duration: 50-500ms
        duration_ms = self.rng.uniform(50, 500)
        duration_samples = int(duration_ms * self.sample_rate / 1000)

        if duration_samples >= batch_size:
            return None

        # Random start time
        start_sample = self.rng.integers(0, batch_size - duration_samples)

        # Affected channels (~30%)
        n_affected = int(self.n_channels * 0.3)
        affected_channels = self.rng.choice(
            self.n_channels, size=n_affected, replace=False
        )

        # Amplitude: 5-20x baseline
        amplitude = self.rng.uniform(5, 20) * baseline_noise

        # Generate EMG-like noise (band-limited)
        emg_noise = self.rng.standard_normal((duration_samples, n_affected))

        # Simple envelope (ramp up, sustain, ramp down)
        envelope = np.ones(duration_samples)
        ramp_len = min(int(0.1 * duration_samples), 20)
        envelope[:ramp_len] = np.linspace(0, 1, ramp_len)
        envelope[-ramp_len:] = np.linspace(1, 0, ramp_len)

        # Apply envelope
        artifact = emg_noise * envelope[:, np.newaxis] * amplitude

        # Inject into samples
        samples[start_sample:start_sample + duration_samples, affected_channels] += artifact

        return ArtifactEvent(
            artifact_type=ArtifactType.JAW_CLENCH,
            start_sample=start_sample,
            duration_samples=duration_samples,
            affected_channels=affected_channels,
            amplitude=amplitude,
        )

    def _inject_electrode_drift(
        self,
        samples: np.ndarray,
        baseline_noise: float,
    ) -> Optional[ArtifactEvent]:
        """
        Inject electrode drift artifact (slow DC shift).

        Characteristics:
            - Duration: ~2 seconds (or remaining batch)
            - Channels: ~5% (electrode contact issues)
            - Amplitude: 3-10x baseline
            - Waveform: Slow linear or exponential drift
        """
        batch_size = samples.shape[0]

        # Duration: ~2 seconds or remaining batch
        duration_samples = min(int(2.0 * self.sample_rate), batch_size)

        # Random start time
        max_start = batch_size - duration_samples
        if max_start <= 0:
            start_sample = 0
            duration_samples = batch_size
        else:
            start_sample = self.rng.integers(0, max_start)

        # Affected channels (~5%)
        n_affected = max(1, int(self.n_channels * 0.05))
        affected_channels = self.rng.choice(
            self.n_channels, size=n_affected, replace=False
        )

        # Amplitude: 3-10x baseline
        amplitude = self.rng.uniform(3, 10) * baseline_noise

        # Generate drift (exponential approach to offset)
        t = np.arange(duration_samples) / duration_samples
        drift = amplitude * (1 - np.exp(-3 * t))  # Exponential rise

        # Random direction per channel
        directions = self.rng.choice([-1, 1], size=n_affected)
        drift = drift[:, np.newaxis] * directions

        # Inject into samples
        samples[start_sample:start_sample + duration_samples, affected_channels] += drift

        return ArtifactEvent(
            artifact_type=ArtifactType.ELECTRODE_DRIFT,
            start_sample=start_sample,
            duration_samples=duration_samples,
            affected_channels=affected_channels,
            amplitude=amplitude,
        )

    def _inject_motion_spike(
        self,
        samples: np.ndarray,
        baseline_noise: float,
    ) -> Optional[ArtifactEvent]:
        """
        Inject motion spike artifact (sharp transient).

        Characteristics:
            - Duration: ~10ms
            - Channels: ~80% (mechanical coupling)
            - Amplitude: 10-50x baseline
            - Waveform: Sharp biphasic spike
        """
        batch_size = samples.shape[0]

        # Duration: ~10ms
        duration_samples = int(0.01 * self.sample_rate)

        if duration_samples >= batch_size:
            return None

        # Random start time
        start_sample = self.rng.integers(0, batch_size - duration_samples)

        # Affected channels (~80%)
        n_affected = int(self.n_channels * 0.8)
        affected_channels = self.rng.choice(
            self.n_channels, size=n_affected, replace=False
        )

        # Amplitude: 10-50x baseline
        amplitude = self.rng.uniform(10, 50) * baseline_noise

        # Generate sharp biphasic spike
        t = np.arange(duration_samples) / duration_samples
        spike = amplitude * np.sin(2 * np.pi * t) * np.exp(-5 * t)

        # Slight amplitude variation per channel
        channel_scale = self.rng.uniform(0.8, 1.2, size=n_affected)
        artifact = spike[:, np.newaxis] * channel_scale

        # Inject into samples
        samples[start_sample:start_sample + duration_samples, affected_channels] += artifact

        return ArtifactEvent(
            artifact_type=ArtifactType.MOTION_SPIKE,
            start_sample=start_sample,
            duration_samples=duration_samples,
            affected_channels=affected_channels,
            amplitude=amplitude,
        )

    def get_artifact_history(self) -> list:
        """Get all injected artifacts."""
        return self._artifact_history.copy()

    def clear_history(self) -> None:
        """Clear artifact history."""
        self._artifact_history.clear()
