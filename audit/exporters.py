"""
Audit Log Exporters for Signal Stability Gateway

Export audit events to JSON and CSV formats for FDA compliance.
"""

import json
import csv
import os
import time
import tempfile
from pathlib import Path
from typing import IO, Literal

from .audit_logger import AuditEvent, AuditLogger


def _write_text_atomically(output_path: str, content: str) -> None:
    """Write text via a temp file to avoid partially written exports."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    handle, temp_path = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(handle, 'w', encoding='utf-8', newline='') as temp_file:
            temp_file.write(content)
        os.replace(temp_path, target)
    except Exception:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as cleanup_error:
                raise RuntimeError(
                    f"Failed to clean up temporary export file {temp_path}"
                ) from cleanup_error
        raise


class JSONExporter:
    """
    Export audit events to JSON format.

    Produces human-readable JSON with proper formatting.
    Suitable for archival and programmatic access.
    """

    @staticmethod
    def export(
        events: list[AuditEvent],
        output_path: str | None = None,
        include_metadata: bool = True,
    ) -> str:
        """
        Export events to JSON.

        Args:
            events: List of AuditEvent to export
            output_path: Optional file path (writes to file if provided)
            include_metadata: Include event metadata fields

        Returns:
            JSON string
        """
        records = []
        for event in events:
            record = event.to_dict()
            if not include_metadata:
                record.pop('metadata', None)
            records.append(record)

        output = {
            'export_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'event_count': len(records),
            'events': records,
        }

        json_str = json.dumps(output, indent=2)

        if output_path:
            _write_text_atomically(output_path, json_str)

        return json_str

    @staticmethod
    def export_logger(
        logger: AuditLogger,
        output_path: str,
        include_counts: bool = True,
    ) -> None:
        """
        Export entire logger state to JSON file.

        Args:
            logger: AuditLogger instance
            output_path: Output file path
            include_counts: Include event count statistics
        """
        events = logger.get_events(limit=100_000)

        output = {
            'export_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'event_count': len(events),
            'events': [e.to_dict() for e in events],
        }

        if include_counts:
            output['statistics'] = logger.get_counts()

        _write_text_atomically(output_path, json.dumps(output, indent=2))


class CSVExporter:
    """
    Export audit events to CSV format.

    Produces flat CSV suitable for spreadsheet analysis.
    Metadata fields are JSON-encoded in a single column.
    """

    # CSV column headers
    COLUMNS = (
        'timestamp_us',
        'timestamp_iso',
        'event_type',
        'severity',
        'message',
        'channel_id',
        'batch_id',
        'metadata',
    )

    @staticmethod
    def export(
        events: list[AuditEvent],
        output_path: str | None = None,
        include_metadata: bool = True,
    ) -> str:
        """
        Export events to CSV.

        Args:
            events: List of AuditEvent to export
            output_path: Optional file path (writes to file if provided)
            include_metadata: Include metadata column

        Returns:
            CSV string
        """
        import io

        output = io.StringIO()
        columns = list(CSVExporter.COLUMNS)
        if not include_metadata:
            columns.remove('metadata')

        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()

        for event in events:
            row = event.to_dict()
            # JSON-encode metadata
            if 'metadata' in row:
                row['metadata'] = json.dumps(row['metadata'])
            if not include_metadata:
                row.pop('metadata', None)
            writer.writerow(row)

        csv_str = output.getvalue()

        if output_path:
            _write_text_atomically(output_path, csv_str)

        return csv_str

    @staticmethod
    def export_logger(
        logger: AuditLogger,
        output_path: str,
    ) -> None:
        """
        Export entire logger state to CSV file.

        Args:
            logger: AuditLogger instance
            output_path: Output file path
        """
        events = logger.get_events(limit=100_000)
        CSVExporter.export(events, output_path=output_path)


class StreamingExporter:
    """
    Stream audit events to file in real-time.

    Appends events as they occur for live logging.
    """

    def __init__(
        self,
        output_path: str,
        format: Literal['jsonl', 'csv'] = 'jsonl',
    ):
        """
        Initialize streaming exporter.

        Args:
            output_path: Output file path
            format: 'jsonl' (JSON Lines) or 'csv'
        """
        self.output_path = Path(output_path)
        self.format = format
        self._file: IO[str] | None = None
        self._csv_writer = None

    def __enter__(self):
        """Open file for streaming."""
        self._file = open(self.output_path, 'a', newline='', encoding='utf-8')

        if self.format == 'csv' and self.output_path.stat().st_size == 0:
            # Write header for new CSV file
            self._csv_writer = csv.DictWriter(
                self._file,
                fieldnames=CSVExporter.COLUMNS
            )
            self._csv_writer.writeheader()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close file."""
        if self._file:
            self._file.close()
            self._file = None

    def write(self, event: AuditEvent) -> None:
        """
        Write single event to stream.

        Args:
            event: AuditEvent to write
        """
        if self._file is None:
            raise RuntimeError("Exporter not opened. Use 'with' statement.")

        record = event.to_dict()

        if self.format == 'jsonl':
            self._file.write(json.dumps(record) + '\n')
        else:
            if self._csv_writer is None:
                self._csv_writer = csv.DictWriter(
                    self._file,
                    fieldnames=CSVExporter.COLUMNS
                )
            record['metadata'] = json.dumps(record['metadata'])
            self._csv_writer.writerow(record)

        self._file.flush()
