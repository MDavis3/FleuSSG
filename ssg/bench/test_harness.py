"""
End-to-end test harness for Signal Stability Gateway.

Runs the full pipeline with synthetic data and optional noise injection.
Validates viability scoring and artifact detection.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..core.constants import (
    BATCH_SIZE,
    N_CHANNELS,
    SAMPLE_RATE_HZ,
    SYNTHETIC_NOISE_AMPLITUDE_UV,
    SYNTHETIC_SPIKE_AMPLITUDE_UV,
)
from ..core.pipeline_runtime import PipelineRuntime, PipelineRuntimeConfig
from ..core.data_types import ChannelMetrics, SanitizedFrame
from ..ingestion.mock_telemetry import MockTelemetryConfig
from ..simulation.noise_models import ArtifactInjectionConfig

MILLISECONDS_PER_SECOND = 1000.0


@dataclass(frozen=True)
class DistributionSummary:
    """Summary statistics for a numeric distribution."""

    min: float
    max: float
    mean: float
    std: float


@dataclass(frozen=True)
class TestResults:
    """Results from a multi-batch test run."""

    total_batches: int
    total_duration_sec: float
    avg_latency_ms: float
    max_latency_ms: float
    min_viable_channels: int
    max_viable_channels: int
    avg_viable_channels: float
    total_artifacts_injected: int
    snr_distribution: DistributionSummary
    isi_distribution: DistributionSummary


@dataclass(frozen=True)
class SingleBatchResult:
    """Result bundle for a single processed batch."""

    samples: np.ndarray
    sanitized_frame: SanitizedFrame
    metrics: ChannelMetrics
    latency_ms: float


@dataclass(frozen=True)
class PerformanceValidationResult:
    """Structured result for CLI-facing latency validation."""

    passed: bool
    avg_latency_ms: float
    max_latency_ms: float
    target_latency_ms: float
    total_batches: int
    batches_per_second: float


class TestHarness:
    """
    End-to-end test harness for the SSG pipeline.

    Runs synthetic data through the full pipeline:
        MockTelemetry -> IngestionEngine -> SanitizationLayer -> ValidationEngine

    Supports:
        - Controlled noise injection
        - Performance profiling
        - Viability validation
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        sample_rate_hz: int = SAMPLE_RATE_HZ,
        batch_size: int = BATCH_SIZE,
        noise_amplitude_uv: float = SYNTHETIC_NOISE_AMPLITUDE_UV,
        spike_amplitude_uv: float = SYNTHETIC_SPIKE_AMPLITUDE_UV,
        seed: int | None = None,
    ):
        """
        Initialize test harness.

        Args:
            n_channels: Number of channels to simulate
            sample_rate_hz: Sampling rate
            batch_size: Batch size for processing
            noise_amplitude_uv: Background noise amplitude
            spike_amplitude_uv: Spike amplitude
            seed: Random seed for reproducibility
        """
        self.n_channels = n_channels
        self.sample_rate = sample_rate_hz
        self.batch_size = batch_size

        self._runtime = PipelineRuntime(
            config=PipelineRuntimeConfig(
                n_channels=n_channels,
                sample_rate_hz=sample_rate_hz,
                batch_size=batch_size,
                telemetry_config=MockTelemetryConfig(
                    noise_amplitude_uv=noise_amplitude_uv,
                    spike_amplitude_uv=spike_amplitude_uv,
                    seed=seed,
                ),
            ),
        )
        self._noise_generator = self._runtime.noise_generator

        self._latencies: list[float] = []
        self._viable_counts: list[int] = []
        self._total_batches = 0

    @staticmethod
    def _artifact_config(multiplier: float) -> ArtifactInjectionConfig:
        """Scale the default artifact rates for a benchmark run."""

        return ArtifactInjectionConfig(
            jaw_clench_prob=0.1 * multiplier,
            electrode_drift_prob=0.05 * multiplier,
            motion_spike_prob=0.02 * multiplier,
            baseline_noise_uv=10.0,
        )

    @staticmethod
    def _summarize_distribution(values: np.ndarray) -> DistributionSummary:
        """Build a typed summary for a NumPy vector."""

        return DistributionSummary(
            min=float(values.min()),
            max=float(values.max()),
            mean=float(values.mean()),
            std=float(values.std()),
        )

    def run(
        self,
        duration_sec: float = 10.0,
        inject_artifacts: bool = True,
        artifact_rate_multiplier: float = 1.0,
        callback: Callable[[ChannelMetrics, float], None] | None = None,
    ) -> TestResults:
        """
        Run an end-to-end test.

        Args:
            duration_sec: Test duration in seconds
            inject_artifacts: Whether to inject biological artifacts
            artifact_rate_multiplier: Multiplier for artifact probabilities
            callback: Optional callback called after each batch with (metrics, latency_ms)

        Returns:
            TestResults with performance and validation metrics
        """
        n_batches = int(duration_sec * self.sample_rate / self.batch_size)
        if n_batches <= 0:
            raise ValueError("duration_sec must cover at least one batch")

        self.reset()
        start_time = time.time()
        artifact_config = self._artifact_config(artifact_rate_multiplier)

        for _ in range(n_batches):
            batch_start = time.perf_counter()
            batch = self._runtime.process_next_batch(
                inject_artifacts=inject_artifacts,
                artifact_config=artifact_config,
            )

            latency_ms = (
                time.perf_counter() - batch_start
            ) * MILLISECONDS_PER_SECOND
            self._latencies.append(latency_ms)
            self._viable_counts.append(batch.metrics.viable_channel_count)
            self._total_batches += 1

            if callback is not None:
                callback(batch.metrics, latency_ms)

        total_duration = time.time() - start_time
        snr_array = batch.metrics.snr
        isi_array = batch.metrics.isi_violation_rate

        return TestResults(
            total_batches=self._total_batches,
            total_duration_sec=total_duration,
            avg_latency_ms=float(np.mean(self._latencies)),
            max_latency_ms=float(np.max(self._latencies)),
            min_viable_channels=int(np.min(self._viable_counts)),
            max_viable_channels=int(np.max(self._viable_counts)),
            avg_viable_channels=float(np.mean(self._viable_counts)),
            total_artifacts_injected=len(self._noise_generator.get_artifact_history()),
            snr_distribution=self._summarize_distribution(snr_array),
            isi_distribution=self._summarize_distribution(isi_array),
        )

    def run_single_batch(
        self,
        inject_artifacts: bool = False,
    ) -> SingleBatchResult:
        """
        Run a single batch through the pipeline.

        Useful for debugging and unit testing.

        Args:
            inject_artifacts: Whether to inject artifacts

        Returns:
            Structured batch result with latency and pipeline outputs
        """
        self.reset()
        batch_start = time.perf_counter()
        batch = self._runtime.process_next_batch(
            inject_artifacts=inject_artifacts,
            artifact_config=ArtifactInjectionConfig(baseline_noise_uv=10.0),
        )
        latency_ms = (time.perf_counter() - batch_start) * MILLISECONDS_PER_SECOND

        return SingleBatchResult(
            samples=batch.samples,
            sanitized_frame=batch.sanitized_frame,
            metrics=batch.metrics,
            latency_ms=latency_ms,
        )

    def validate_performance(
        self,
        target_latency_ms: float = 5.0,
        duration_sec: float = 30.0,
    ) -> PerformanceValidationResult:
        """
        Validate that pipeline performance meets requirements.

        Args:
            target_latency_ms: Maximum acceptable average latency
            duration_sec: Test duration

        Returns:
            Structured pass/fail result with latency details
        """
        results = self.run(duration_sec=duration_sec, inject_artifacts=True)
        passed = results.avg_latency_ms <= target_latency_ms

        return PerformanceValidationResult(
            passed=passed,
            avg_latency_ms=results.avg_latency_ms,
            max_latency_ms=results.max_latency_ms,
            target_latency_ms=target_latency_ms,
            total_batches=results.total_batches,
            batches_per_second=results.total_batches / results.total_duration_sec,
        )

    def reset(self) -> None:
        """Reset all pipeline components and accumulated run state."""

        self._runtime.reset()
        self._latencies.clear()
        self._viable_counts.clear()
        self._total_batches = 0
