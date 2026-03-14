import numpy as np

from ssg.core.data_types import ChannelMetrics, SanitizedFrame
from ssg.core.pipeline_runtime import PipelineDependencies, PipelineRuntime, PipelineRuntimeConfig
from ssg.simulation.noise_models import ArtifactEvent, ArtifactInjectionConfig, ArtifactType


class FakeTelemetry:
    def __init__(self):
        self.reset_called = False
        self.generated = []

    def generate_batch(self, batch_size):
        samples = np.ones((batch_size, 2), dtype=np.float32)
        timestamps = np.arange(batch_size, dtype=np.uint64)
        self.generated.append(batch_size)
        return samples, timestamps

    def reset(self):
        self.reset_called = True


class FakeIngestion:
    def __init__(self):
        self.ingested = None
        self.cleared = False

    def ingest_batch(self, samples, timestamps):
        self.ingested = (samples.copy(), timestamps.copy())

    def clear(self):
        self.cleared = True


class FakeSanitization:
    def __init__(self):
        self.processed = None
        self.reset_called = False

    def sanitize_batch(self, samples, timestamps):
        self.processed = (samples.copy(), timestamps.copy())
        return SanitizedFrame(
            timestamp_us=int(timestamps[0]),
            raw_unfiltered=samples.copy(),
            lfp=samples.copy(),
            spikes=samples.copy(),
            artifact_flags=np.zeros(samples.shape[1], dtype=bool),
        )

    def reset(self):
        self.reset_called = True


class FakeValidation:
    def __init__(self):
        self.processed = None
        self.reset_called = False

    def process(self, sanitized, timestamps):
        self.processed = (sanitized, timestamps.copy())
        viability_mask = np.array([True, False], dtype=bool)
        return ChannelMetrics(
            timestamp_us=sanitized.timestamp_us,
            snr=np.array([5.0, 4.0], dtype=np.float32),
            firing_rate_hz=np.array([10.0, 11.0], dtype=np.float32),
            isi_violation_rate=np.zeros(2, dtype=np.float32),
            impedance_kohm=np.array([500.0, 500.0], dtype=np.float32),
            viability_mask=viability_mask,
            viable_channel_count=int(viability_mask.sum()),
        )

    def reset(self):
        self.reset_called = True


class FakeNoiseGenerator:
    def __init__(self):
        self.injected = []
        self.cleared = False

    def inject_artifacts(self, samples, config=None):
        self.injected.append(config)
        return samples + 1.0, [
            ArtifactEvent(
                artifact_type=ArtifactType.MOTION_SPIKE,
                start_sample=0,
                duration_samples=2,
                affected_channels=np.array([0], dtype=np.uint16),
                amplitude=25.0,
            )
        ]

    def clear_history(self):
        self.cleared = True


def test_pipeline_runtime_processes_batches_and_resets_dependencies():
    telemetry = FakeTelemetry()
    ingestion = FakeIngestion()
    sanitization = FakeSanitization()
    validation = FakeValidation()
    noise = FakeNoiseGenerator()
    runtime = PipelineRuntime(
        config=PipelineRuntimeConfig(n_channels=2, sample_rate_hz=1000, batch_size=4),
        dependencies=PipelineDependencies(
            telemetry=telemetry,
            ingestion=ingestion,
            sanitization=sanitization,
            validation=validation,
            noise_generator=noise,
        ),
    )

    result = runtime.process_next_batch(
        inject_artifacts=True,
        artifact_config=ArtifactInjectionConfig(baseline_noise_uv=25.0),
    )

    assert telemetry.generated == [4]
    assert ingestion.ingested[0].shape == (4, 2)
    assert sanitization.processed[0][0, 0] == 2.0
    assert validation.processed[0].timestamp_us == 0
    assert result.samples.shape == (4, 2)
    assert result.metrics.viable_channel_count == 1
    assert len(result.artifacts) == 1
    assert result.artifacts[0].artifact_type is ArtifactType.MOTION_SPIKE
    assert noise.injected[0].baseline_noise_uv == 25.0
    assert runtime.telemetry is telemetry
    assert runtime.ingestion is ingestion
    assert runtime.sanitization is sanitization
    assert runtime.validation is validation
    assert runtime.noise_generator is noise

    runtime.reset()

    assert telemetry.reset_called
    assert ingestion.cleared
    assert sanitization.reset_called
    assert validation.reset_called
    assert noise.cleared