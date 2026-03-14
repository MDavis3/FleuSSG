from ssg.audit.audit_logger import AuditEventContext, AuditLogger
from ssg.audit.event_types import EventSeverity, EventType


def test_audit_logger_records_context_and_counts():
    logger = AuditLogger()

    event = logger.log(
        EventType.SYSTEM_START,
        "session started",
        severity=EventSeverity.INFO,
        context=AuditEventContext(
            batch_id=3,
            metadata={"source": "test"},
        ),
    )

    assert event is not None
    assert event.batch_id == 3
    assert event.metadata == {"source": "test"}

    counts = logger.get_counts()
    assert counts["total_events"] == 1
    assert counts["by_type"]["system_start"] == 1
    assert counts["by_severity"]["INFO"] == 1
