"""
End-to-End Test Harness for Signal Stability Gateway

Runs the full pipeline with synthetic data and noise injection.
Validates viability scoring and artifact detection.
"""

import time
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass

from ..core.constants import (
    N_CHANNELS,
    SAMPLE_RATE_HZ,
    BATCH_SIZE,
    SYNTHETIC_NOISE_AMPLITUDE_UV,
    SYNTHETIC_SPIKE_AMPLITUDE_UV,
)
from ..ingestion.mock_telemetry import MockTelemetry
from ..ingestion.engine import IngestionEngine
from ..sanitization.layer import SanitizationLayer
from ..validation.engine import ValidationEngine
from ..core.data_types import ChannelMetrics
from .noise_models import NoiseGenerator


@dataclass
class TestResults:
    """Results from a test run."""
    total_batches: int
    total_duration_sec: float
    avg_latency_ms: float
    max_latency_ms: float
    min_viable_channels: int
    max_viable_channels: int
    avg_viable_channels: float
    total_artifacts_injected: int
    snr_distribution: dict
    isi_distribution: dict


class TestHarness:
    """
    End-to-end test harness for SSG pipeline.

    Runs synthetic data through the full pipeline:
        MockTelemetry → IngestionEngine → SanitizationLayer → ValidationEngine

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
        seed: Optional[int] = None,
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

        # Initialize pipeline components
        self._telemetry = MockTelemetry(
            n_channels=n_channels,
            sample_rate_hz=sample_rate_hz,
            noise_amplitude_uv=noise_amplitude_uv,
            spike_amplitude_uv=spike_amplitude_uv,
            seed=seed,
        )

        self._ingestion = IngestionEngine(
            n_channels=n_channels,
            sample_rate_hz=sample_rate_hz,
        )

        self._sanitization = SanitizationLayer(
            n_channels=n_channels,
            sample_rate_hz=sample_rate_hz,
        )

        self._validation = ValidationEngine(
            n_channels=n_channels,
            sample_rate_hz=sample_rate_hz,
        )

        self._noise_generator = NoiseGenerator(
            n_channels=n_channels,
            sample_rate_hz=sample_rate_hz,
            seed=seed,
        )

        # Results tracking
        self._latencies: list = []
        self._viable_counts: list = []
        self._total_batches = 0

    def run(
        self,
        duration_sec: float = 10.0,
        inject_artifacts: bool = True,
        artifact_rate_multiplier: float = 1.0,
        callback: Optional[Callable[[ChannelMetrics, float], None]] = None,
    ) -> TestResults:
        """
        Run end-to-end test.

        Args:
            duration_sec: Test duration in seconds
            inject_artifacts: Whether to inject biological artifacts
            artifact_rate_multiplier: Multiplier for artifact probabilities
            callback: Optional callback called after each batch with (metrics, latency_ms)

        Returns:
            TestResults with performance and validation metrics
        """
        n_batches = int(duration_sec * self.sample_rate / self.batch_size)

        self._latencies.clear()
        self._viable_counts.clear()
        self._total_batches = 0

        start_time = time.time()

        for batch_idx in range(n_batches):
            batch_start = time.perf_counter()

            # Generate synthetic data
            samples, timestamps = self._telemetry.generate_batch(self.batch_size)

            # Inject artifacts if enabled
            if inject_artifacts:
                samples, _ = self._noise_generator.inject_artifacts(
                    samples,
                    jaw_clench_prob=0.1 * artifact_rate_multiplier,
                    electrode_drift_prob=0.05 * artifact_rate_multiplier,
                    motion_spike_prob=0.02 * artifact_rate_multiplier,
                    baseline_noise=10.0,
                )

            # Run through pipeline
            self._ingestion.ingest_batch(samples, timestamps)

            sanitized = self._sanitization.process(samples, timestamps)

            metrics = self._validation.process(sanitized, timestamps)

            batch_end = time.perf_counter()
            latency_ms = (batch_end - batch_start) * 1000

            self._latencies.append(latency_ms)
            self._viable_counts.append(metrics.viable_channel_count)
            self._total_batches += 1

            if callback:
                callback(metrics, latency_ms)

        total_duration = time.time() - start_time

        # Compute statistics
        snr_array = metrics.snr
        isi_array = metrics.isi_violation_rate

        return TestResults(
            total_batches=self._total_batches,
            total_duration_sec=total_duration,
            avg_latency_ms=np.mean(self._latencies),
            max_latency_ms=np.max(self._latencies),
            min_viable_channels=np.min(self._viable_counts),
            max_viable_channels=np.max(self._viable_counts),
            avg_viable_channels=np.mean(self._viable_counts),
            total_artifacts_injected=len(self._noise_generator.get_artifact_history()),
            snr_distribution={
                'min': float(snr_array.min()),
                'max': float(snr_array.max()),
                'mean': float(snr_array.mean()),
                'std': float(snr_array.std()),
            },
            isi_distribution={
                'min': float(isi_array.min()),
                'max': float(isi_array.max()),
                'mean': float(isi_array.mean()),
                'std': float(isi_array.std()),
            },
        )

    def run_single_batch(
        self,
        inject_artifacts: bool = False,
    ) -> tuple:
        """
        Run a single batch through the pipeline.

        Useful for debugging and unit testing.

        Args:
            inject_artifacts: Whether to inject artifacts

        Returns:
            Tuple of (samples, sanitized_frame, metrics, latency_ms)
        """
        batch_start = time.perf_counter()

        # Generate data
        samples, timestamps = self._telemetry.generate_batch(self.batch_size)

        if inject_artifacts:
            samples, _ = self._noise_generator.inject_artifacts(
                samples, baseline_noise=10.0
            )

        # Process
        self._ingestion.ingest_batch(samples, timestamps)
        sanitized = self._sanitization.process(samples, timestamps)
        metrics = self._validation.process(sanitized, timestamps)

        latency_ms = (time.perf_counter() - batch_start) * 1000

        return samples, sanitized, metrics, latency_ms

    def validate_performance(
        self,
        target_latency_ms: float = 5.0,
        duration_sec: float = 30.0,
    ) -> dict:
        """
        Validate pipeline performance meets requirements.

        Args:
            target_latency_ms: Maximum acceptable average latency
            duration_sec: Test duration

        Returns:
            Dict with pass/fail status and details
        """
        results = self.run(duration_sec=duration_sec, inject_artifacts=True)

        passed = results.avg_latency_ms <= target_latency_ms

        return {
            'passed': passed,
            'avg_latency_ms': results.avg_latency_ms,
            'max_latency_ms': results.max_latency_ms,
            'target_latency_ms': target_latency_ms,
            'total_batches': results.total_batches,
            'batches_per_second': results.total_batches / results.total_duration_sec,
        }

    def reset(self) -> None:
        """Reset all pipeline components."""
        self._telemetry.reset()
        self._ingestion.clear()
        self._sanitization.reset()
        self._validation.reset()
        self._noise_generator.clear_history()
        self._latencies.clear()
        self._viable_counts.clear()
        self._total_batches = 0
