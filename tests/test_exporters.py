from ssg.audit.audit_logger import AuditEventContext, AuditLogger
from ssg.audit.event_types import EventType
from ssg.audit.exporters import CSVExporter, JSONExporter


def test_exporters_write_utf8_text_files(tmp_path):
    logger = AuditLogger()
    logger.log(
        EventType.SYSTEM_START,
        "calibrated",
        context=AuditEventContext(metadata={"operator": "unit-test"}),
    )

    json_path = tmp_path / "audit.json"
    csv_path = tmp_path / "audit.csv"

    JSONExporter.export_logger(logger, str(json_path))
    CSVExporter.export_logger(logger, str(csv_path))

    json_text = json_path.read_text(encoding="utf-8")
    csv_text = csv_path.read_text(encoding="utf-8")

    assert '"event_count": 1' in json_text
    assert "system_start" in json_text
    assert "system_start" in csv_text
    assert "operator" in csv_text
