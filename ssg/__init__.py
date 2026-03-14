"""Signal Stability Gateway public package surface."""

from .audit.audit_logger import AuditEvent, AuditEventContext, AuditLogger
from .audit.event_types import EventSeverity, EventType
from .audit.exporters import CSVExporter, JSONExporter, StreamingExporter
from .bench.test_harness import (
    DistributionSummary,
    PerformanceValidationResult,
    SingleBatchResult,
    TestHarness,
    TestResults,
)
from .core.array_types import (
    BoolVector,
    ChannelIndexVector,
    FloatMatrix,
    FloatVector,
    TimestampVector,
)
from .core.constants import BATCH_DURATION_SEC, BATCH_SIZE, N_CHANNELS, SAMPLE_RATE_HZ
from .core.data_types import (
    AuditEvent as AuditPayload,
    ChannelFrameDtype,
    ChannelMetrics,
    RegionMetrics,
    SanitizedFrame,
    make_channel_frame_dtype,
)
from .core.pipeline_runtime import (
    PipelineBatchResult,
    PipelineDependencies,
    PipelineRuntime,
    PipelineRuntimeConfig,
)
from .core.ring_buffer import RingBuffer
from .dashboard.cli import Dashboard, DashboardEvent
from .ingestion.engine import IngestionEngine
from .ingestion.mock_telemetry import (
    MockTelemetry,
    MockTelemetryConfig,
    RealtimeMockTelemetry,
)
from .sanitization.layer import SOSFilterState, SanitizationLayer
from .simulation.noise_models import (
    ArtifactEvent,
    ArtifactInjectionConfig,
    ArtifactType,
    NoiseGenerator,
)
from .validation.engine import ValidationEngine

__version__ = "0.1.0"
__author__ = "Signal Yield"

__all__ = [
    "__author__",
    "__version__",
    "AuditEvent",
    "AuditEventContext",
    "AuditLogger",
    "AuditPayload",
    "ArtifactEvent",
    "ArtifactInjectionConfig",
    "ArtifactType",
    "BATCH_DURATION_SEC",
    "BATCH_SIZE",
    "BoolVector",
    "CSVExporter",
    "ChannelFrameDtype",
    "ChannelIndexVector",
    "ChannelMetrics",
    "Dashboard",
    "DashboardEvent",
    "DistributionSummary",
    "EventSeverity",
    "EventType",
    "FloatMatrix",
    "FloatVector",
    "IngestionEngine",
    "JSONExporter",
    "MockTelemetry",
    "MockTelemetryConfig",
    "N_CHANNELS",
    "NoiseGenerator",
    "PerformanceValidationResult",
    "PipelineBatchResult",
    "PipelineDependencies",
    "PipelineRuntime",
    "PipelineRuntimeConfig",
    "RealtimeMockTelemetry",
    "RegionMetrics",
    "RingBuffer",
    "SAMPLE_RATE_HZ",
    "SOSFilterState",
    "SanitizationLayer",
    "SanitizedFrame",
    "SingleBatchResult",
    "StreamingExporter",
    "TestHarness",
    "TestResults",
    "TimestampVector",
    "ValidationEngine",
    "make_channel_frame_dtype",
]
