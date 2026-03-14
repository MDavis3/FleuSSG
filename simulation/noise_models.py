"""
Biological Noise Models for SSG Testing

Simulates realistic artifacts for validation testing:
- Jaw Clench: High-amplitude EMG burst
- Electrode Drift: Slow DC shift
- Motion Spike: Sharp transient

All models are parameterized based on clinical observations.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum, auto

from ..core.array_types import ChannelIndexVector, FloatMatrix
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


@dataclass(frozen=True)
class ArtifactEvent:
    """Record of an injected artifact."""

    artifact_type: ArtifactType
    start_sample: int
    duration_samples: int
    affected_channels: ChannelIndexVector
    amplitude: float


@dataclass(frozen=True)
class ArtifactInjectionConfig:
    """Parameters controlling probabilistic artifact injection."""

    jaw_clench_prob: float = JAW_CLENCH_PROBABILITY
    electrode_drift_prob: float = ELECTRODE_DRIFT_PROBABILITY
    motion_spike_prob: float = MOTION_SPIKE_PROBABILITY
    baseline_noise_uv: float = 10.0


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
        seed: int | None = None,
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
        self._artifact_history: list[ArtifactEvent] = []

    def inject_artifacts(
        self,
        samples: FloatMatrix,
        config: ArtifactInjectionConfig | None = None,
    ) -> tuple[FloatMatrix, list[ArtifactEvent]]:
        """
        Inject biological artifacts into signal.

        Artifacts are applied in-place for efficiency.

        Args:
            samples: Input signal, shape (batch_size, n_channels)
            config: Artifact injection probabilities and amplitudes

        Returns:
            Tuple of (modified samples, list of ArtifactEvent)
        """
        injection_config = config or ArtifactInjectionConfig()
        batch_size = samples.shape[0]
        batch_duration_sec = batch_size / self.sample_rate
        injected_artifacts: list[ArtifactEvent] = []

        n_jaw_clench = self.rng.poisson(
            injection_config.jaw_clench_prob * batch_duration_sec
        )
        n_electrode_drift = self.rng.poisson(
            injection_config.electrode_drift_prob * batch_duration_sec
        )
        n_motion_spike = self.rng.poisson(
            injection_config.motion_spike_prob * batch_duration_sec
        )

        for _ in range(n_jaw_clench):
            artifact = self._inject_jaw_clench(
                samples,
                injection_config.baseline_noise_uv,
            )
            if artifact is not None:
                injected_artifacts.append(artifact)

        for _ in range(n_electrode_drift):
            artifact = self._inject_electrode_drift(
                samples,
                injection_config.baseline_noise_uv,
            )
            if artifact is not None:
                injected_artifacts.append(artifact)

        for _ in range(n_motion_spike):
            artifact = self._inject_motion_spike(
                samples,
                injection_config.baseline_noise_uv,
            )
            if artifact is not None:
                injected_artifacts.append(artifact)

        self._artifact_history.extend(injected_artifacts)
        return samples, injected_artifacts

    def _inject_jaw_clench(
        self,
        samples: FloatMatrix,
        baseline_noise_uv: float,
    ) -> ArtifactEvent | None:
        """
        Inject jaw clench artifact (EMG burst).

        Characteristics:
            - Duration: 50-500ms
            - Channels: ~30% (random subset)
            - Amplitude: 5-20x baseline
            - Waveform: Band-limited noise (100-500Hz)
        """
        batch_size = samples.shape[0]
        duration_ms = self.rng.uniform(50, 500)
        duration_samples = int(duration_ms * self.sample_rate / 1000)

        if duration_samples >= batch_size:
            return None

        start_sample = int(self.rng.integers(0, batch_size - duration_samples))
        n_affected = int(self.n_channels * 0.3)
        affected_channels = self.rng.choice(
            self.n_channels,
            size=n_affected,
            replace=False,
        ).astype(np.uint16)
        amplitude = float(self.rng.uniform(5, 20) * baseline_noise_uv)

        emg_noise = self.rng.standard_normal((duration_samples, n_affected))
        envelope = np.ones(duration_samples)
        ramp_len = min(int(0.1 * duration_samples), 20)
        envelope[:ramp_len] = np.linspace(0, 1, ramp_len)
        envelope[-ramp_len:] = np.linspace(1, 0, ramp_len)
        artifact = emg_noise * envelope[:, np.newaxis] * amplitude

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
        samples: FloatMatrix,
        baseline_noise_uv: float,
    ) -> ArtifactEvent | None:
        """
        Inject electrode drift artifact (slow DC shift).

        Characteristics:
            - Duration: ~2 seconds (or remaining batch)
            - Channels: ~5% (electrode contact issues)
            - Amplitude: 3-10x baseline
            - Waveform: Slow linear or exponential drift
        """
        batch_size = samples.shape[0]
        duration_samples = min(int(2.0 * self.sample_rate), batch_size)

        max_start = batch_size - duration_samples
        if max_start <= 0:
            start_sample = 0
            duration_samples = batch_size
        else:
            start_sample = int(self.rng.integers(0, max_start))

        n_affected = max(1, int(self.n_channels * 0.05))
        affected_channels = self.rng.choice(
            self.n_channels,
            size=n_affected,
            replace=False,
        ).astype(np.uint16)
        amplitude = float(self.rng.uniform(3, 10) * baseline_noise_uv)

        t = np.arange(duration_samples) / duration_samples
        drift = amplitude * (1 - np.exp(-3 * t))
        directions = self.rng.choice([-1, 1], size=n_affected)
        drift = drift[:, np.newaxis] * directions

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
        samples: FloatMatrix,
        baseline_noise_uv: float,
    ) -> ArtifactEvent | None:
        """
        Inject motion spike artifact (sharp transient).

        Characteristics:
            - Duration: ~10ms
            - Channels: ~80% (mechanical coupling)
            - Amplitude: 10-50x baseline
            - Waveform: Sharp biphasic spike
        """
        batch_size = samples.shape[0]
        duration_samples = int(0.01 * self.sample_rate)

        if duration_samples >= batch_size:
            return None

        start_sample = int(self.rng.integers(0, batch_size - duration_samples))
        n_affected = int(self.n_channels * 0.8)
        affected_channels = self.rng.choice(
            self.n_channels,
            size=n_affected,
            replace=False,
        ).astype(np.uint16)
        amplitude = float(self.rng.uniform(10, 50) * baseline_noise_uv)

        t = np.arange(duration_samples) / duration_samples
        spike = amplitude * np.sin(2 * np.pi * t) * np.exp(-5 * t)
        channel_scale = self.rng.uniform(0.8, 1.2, size=n_affected)
        artifact = spike[:, np.newaxis] * channel_scale

        samples[start_sample:start_sample + duration_samples, affected_channels] += artifact

        return ArtifactEvent(
            artifact_type=ArtifactType.MOTION_SPIKE,
            start_sample=start_sample,
            duration_samples=duration_samples,
            affected_channels=affected_channels,
            amplitude=amplitude,
        )

    def get_artifact_history(self) -> list[ArtifactEvent]:
        """Get all injected artifacts."""
        return self._artifact_history.copy()

    def clear_history(self) -> None:
        """Clear artifact history."""
        self._artifact_history.clear()
