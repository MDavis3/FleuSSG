from ssg import (
    AuditLogger,
    Dashboard,
    IngestionEngine,
    JSONExporter,
    MockTelemetry,
    N_CHANNELS,
    PipelineRuntime,
    SanitizationLayer,
    TestHarness as RuntimeHarness,
    ValidationEngine,
)


def test_root_package_exports_stable_public_api():
    assert AuditLogger.__name__ == "AuditLogger"
    assert Dashboard.__name__ == "Dashboard"
    assert IngestionEngine.__name__ == "IngestionEngine"
    assert JSONExporter.__name__ == "JSONExporter"
    assert MockTelemetry.__name__ == "MockTelemetry"
    assert PipelineRuntime.__name__ == "PipelineRuntime"
    assert SanitizationLayer.__name__ == "SanitizationLayer"
    assert RuntimeHarness.__name__ == "TestHarness"
    assert ValidationEngine.__name__ == "ValidationEngine"
    assert N_CHANNELS > 0
