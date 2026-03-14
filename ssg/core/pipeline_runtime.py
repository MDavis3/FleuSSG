"""Shared runtime helpers for assembling and advancing the SSG pipeline."""

from dataclasses import dataclass, field

from .array_types import FloatMatrix, TimestampVector
from .constants import BATCH_SIZE, N_CHANNELS, SAMPLE_RATE_HZ
from .data_types import ChannelMetrics, SanitizedFrame
from ..ingestion.engine import IngestionEngine
from ..ingestion.mock_telemetry import MockTelemetry, MockTelemetryConfig
from ..sanitization.layer import SanitizationLayer
from ..simulation.noise_models import (
    ArtifactEvent,
    ArtifactInjectionConfig,
    NoiseGenerator,
)
from ..validation.engine import ValidationEngine


@dataclass(frozen=True)
class PipelineRuntimeConfig:
    """Configuration for a reusable telemetry-to-metrics runtime."""

    n_channels: int = N_CHANNELS
    sample_rate_hz: int = SAMPLE_RATE_HZ
    batch_size: int = BATCH_SIZE
    telemetry_config: MockTelemetryConfig = field(default_factory=MockTelemetryConfig)


@dataclass
class PipelineDependencies:
    """Optional collaborators for a pipeline runtime."""

    telemetry: MockTelemetry | None = None
    ingestion: IngestionEngine | None = None
    sanitization: SanitizationLayer | None = None
    validation: ValidationEngine | None = None
    noise_generator: NoiseGenerator | None = None


@dataclass(frozen=True)
class PipelineBatchResult:
    """End-to-end output for a single processed telemetry batch."""

    samples: FloatMatrix
    timestamps: TimestampVector
    sanitized_frame: SanitizedFrame
    metrics: ChannelMetrics
    artifacts: tuple[ArtifactEvent, ...] = ()


class PipelineRuntime:
    """Own the pipeline components and advance them one batch at a time."""

    def __init__(
        self,
        config: PipelineRuntimeConfig | None = None,
        dependencies: PipelineDependencies | None = None,
    ):
        self.config = config or PipelineRuntimeConfig()
        runtime_dependencies = dependencies or PipelineDependencies()

        self._telemetry = runtime_dependencies.telemetry or MockTelemetry(
            n_channels=self.config.n_channels,
            sample_rate_hz=self.config.sample_rate_hz,
            config=self.config.telemetry_config,
        )
        self._ingestion = runtime_dependencies.ingestion or IngestionEngine(
            n_channels=self.config.n_channels,
            sample_rate_hz=self.config.sample_rate_hz,
        )
        self._sanitization = runtime_dependencies.sanitization or SanitizationLayer(
            n_channels=self.config.n_channels,
            sample_rate_hz=self.config.sample_rate_hz,
        )
        self._validation = runtime_dependencies.validation or ValidationEngine(
            n_channels=self.config.n_channels,
            sample_rate_hz=self.config.sample_rate_hz,
        )
        self._noise_generator = runtime_dependencies.noise_generator or NoiseGenerator(
            n_channels=self.config.n_channels,
            sample_rate_hz=self.config.sample_rate_hz,
            seed=self.config.telemetry_config.seed,
        )

    @property
    def telemetry(self) -> MockTelemetry:
        """Return the telemetry source."""
        return self._telemetry

    @property
    def ingestion(self) -> IngestionEngine:
        """Return the ingestion stage."""
        return self._ingestion

    @property
    def sanitization(self) -> SanitizationLayer:
        """Return the sanitization stage."""
        return self._sanitization

    @property
    def validation(self) -> ValidationEngine:
        """Return the validation stage."""
        return self._validation

    @property
    def noise_generator(self) -> NoiseGenerator:
        """Return the artifact injector."""
        return self._noise_generator

    def process_next_batch(
        self,
        inject_artifacts: bool = False,
        artifact_config: ArtifactInjectionConfig | None = None,
    ) -> PipelineBatchResult:
        """Generate, ingest, sanitize, and validate one batch."""

        samples, timestamps = self._telemetry.generate_batch(self.config.batch_size)
        artifacts: tuple[ArtifactEvent, ...] = ()
        if inject_artifacts:
            samples, injected = self._noise_generator.inject_artifacts(
                samples,
                config=artifact_config,
            )
            artifacts = tuple(injected)

        self._ingestion.ingest_batch(samples, timestamps)
        sanitized = self._sanitization.sanitize_batch(samples, timestamps)
        metrics = self._validation.process(sanitized, timestamps)
        return PipelineBatchResult(
            samples=samples,
            timestamps=timestamps,
            sanitized_frame=sanitized,
            metrics=metrics,
            artifacts=artifacts,
        )

    def reset(self) -> None:
        """Reset all collaborators to a clean run state."""

        self._telemetry.reset()
        self._ingestion.clear()
        self._sanitization.reset()
        self._validation.reset()
        self._noise_generator.clear_history()
