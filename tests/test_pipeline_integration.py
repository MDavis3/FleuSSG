from ssg.bench.test_harness import TestHarness as PipelineHarness
from ssg.audit.audit_logger import AuditLogger
from ssg.audit.event_types import EventSeverity, EventType
from ssg.audit.exporters import JSONExporter


def test_pipeline_harness_and_exporters_work_together(tmp_path):
    harness = PipelineHarness(seed=7)
    batch = harness.run_single_batch(inject_artifacts=True)

    logger = AuditLogger(min_severity=EventSeverity.DEBUG)
    logger.log_batch(
        batch_id=0,
        viable_count=batch.metrics.viable_channel_count,
        latency_ms=batch.latency_ms,
        artifacts_detected=int(batch.sanitized_frame.artifact_flags.sum()),
    )
    logger.log(EventType.SYSTEM_STOP, "completed integration test")

    output_path = tmp_path / "integration-log.json"
    JSONExporter.export_logger(logger, str(output_path))

    text = output_path.read_text(encoding="utf-8")
    assert batch.metrics.viable_channel_count >= 0
    assert '"event_count": 2' in text
    assert "data_batch_processed" in text


def test_pipeline_harness_resets_artifact_history_between_runs():
    harness = PipelineHarness(seed=11)

    first = harness.run(
        duration_sec=0.1,
        inject_artifacts=True,
        artifact_rate_multiplier=1000.0,
    )
    second = harness.run(duration_sec=0.1, inject_artifacts=False)

    assert first.total_artifacts_injected > 0
    assert second.total_artifacts_injected == 0
