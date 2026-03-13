"""
FDA Audit Logger for Signal Stability Gateway

Provides regulatory-compliant logging with:
- Timestamped events
- Event enumeration
- Export to JSON/CSV
- Configurable retention
"""

import time
import threading
from typing import Optional, List, Dict, Any
from collections import deque
from dataclasses import dataclass, field, asdict

from .event_types import EventType, EventSeverity


@dataclass
class AuditEvent:
    """
    Single audit event record.

    Immutable after creation for regulatory compliance.
    """
    timestamp_us: int
    event_type: EventType
    severity: EventSeverity
    message: str
    channel_id: Optional[int] = None
    batch_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for export."""
        return {
            'timestamp_us': self.timestamp_us,
            'timestamp_iso': self._format_timestamp(),
            'event_type': self.event_type.value,
            'severity': self.severity.name,
            'message': self.message,
            'channel_id': self.channel_id,
            'batch_id': self.batch_id,
            'metadata': self.metadata,
        }

    def _format_timestamp(self) -> str:
        """Format timestamp as ISO 8601."""
        seconds = self.timestamp_us / 1_000_000
        return time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(seconds))


class AuditLogger:
    """
    Thread-safe FDA audit logger.

    Features:
        - Configurable maximum event retention
        - Severity filtering
        - Event counting by type
        - Export capabilities

    Usage:
        logger = AuditLogger()
        logger.log(EventType.SYSTEM_START, "SSG initialized")
        logger.log_channel_event(EventType.CHANNEL_SNR_LOW, channel_id=42)
    """

    def __init__(
        self,
        max_events: int = 100_000,
        min_severity: EventSeverity = EventSeverity.INFO,
    ):
        """
        Initialize audit logger.

        Args:
            max_events: Maximum events to retain in memory
            min_severity: Minimum severity level to log
        """
        self._events: deque = deque(maxlen=max_events)
        self._min_severity = min_severity
        self._lock = threading.Lock()

        # Event counters
        self._event_counts: Dict[EventType, int] = {}
        self._severity_counts: Dict[EventSeverity, int] = {}

        # Session tracking
        self._session_start_us = int(time.time() * 1_000_000)
        self._batch_counter = 0

    def log(
        self,
        event_type: EventType,
        message: str,
        severity: EventSeverity = EventSeverity.INFO,
        channel_id: Optional[int] = None,
        batch_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[AuditEvent]:
        """
        Log an audit event.

        Args:
            event_type: Type of event
            message: Human-readable description
            severity: Event severity
            channel_id: Associated channel (if applicable)
            batch_id: Associated batch (if applicable)
            metadata: Additional key-value data

        Returns:
            Created AuditEvent if logged, None if filtered
        """
        # Filter by severity
        if severity.value < self._min_severity.value:
            return None

        timestamp_us = int(time.time() * 1_000_000)

        event = AuditEvent(
            timestamp_us=timestamp_us,
            event_type=event_type,
            severity=severity,
            message=message,
            channel_id=channel_id,
            batch_id=batch_id,
            metadata=metadata or {},
        )

        with self._lock:
            self._events.append(event)

            # Update counters
            self._event_counts[event_type] = self._event_counts.get(event_type, 0) + 1
            self._severity_counts[severity] = self._severity_counts.get(severity, 0) + 1

        return event

    def log_channel_event(
        self,
        event_type: EventType,
        channel_id: int,
        message: Optional[str] = None,
        severity: EventSeverity = EventSeverity.INFO,
        **metadata,
    ) -> Optional[AuditEvent]:
        """
        Log a channel-specific event.

        Convenience method for per-channel events.
        """
        if message is None:
            message = f"Channel {channel_id}: {event_type.value}"

        return self.log(
            event_type=event_type,
            message=message,
            severity=severity,
            channel_id=channel_id,
            metadata=metadata,
        )

    def log_artifact(
        self,
        event_type: EventType,
        affected_channels: int,
        amplitude: float,
        duration_ms: float,
        batch_id: Optional[int] = None,
    ) -> Optional[AuditEvent]:
        """
        Log an artifact detection event.

        Args:
            event_type: Type of artifact
            affected_channels: Number of channels affected
            amplitude: Artifact amplitude relative to baseline
            duration_ms: Duration in milliseconds
            batch_id: Batch where artifact was detected
        """
        return self.log(
            event_type=event_type,
            message=f"Artifact: {affected_channels} channels, {amplitude:.1f}x amplitude, {duration_ms:.1f}ms",
            severity=EventSeverity.WARNING,
            batch_id=batch_id,
            metadata={
                'affected_channels': affected_channels,
                'amplitude': amplitude,
                'duration_ms': duration_ms,
            },
        )

    def log_batch(
        self,
        batch_id: int,
        viable_count: int,
        latency_ms: float,
        artifacts_detected: int = 0,
    ) -> Optional[AuditEvent]:
        """
        Log batch processing completion.

        Debug-level event for detailed tracing.
        """
        self._batch_counter = batch_id

        return self.log(
            event_type=EventType.DATA_BATCH_PROCESSED,
            message=f"Batch {batch_id}: {viable_count} viable, {latency_ms:.1f}ms latency",
            severity=EventSeverity.DEBUG,
            batch_id=batch_id,
            metadata={
                'viable_count': viable_count,
                'latency_ms': latency_ms,
                'artifacts_detected': artifacts_detected,
            },
        )

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        min_severity: Optional[EventSeverity] = None,
        channel_id: Optional[int] = None,
        limit: int = 1000,
    ) -> List[AuditEvent]:
        """
        Query logged events with filters.

        Args:
            event_type: Filter by event type
            min_severity: Filter by minimum severity
            channel_id: Filter by channel
            limit: Maximum events to return

        Returns:
            List of matching events (newest first)
        """
        with self._lock:
            events = list(self._events)

        # Apply filters
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]

        if min_severity is not None:
            events = [e for e in events if e.severity.value >= min_severity.value]

        if channel_id is not None:
            events = [e for e in events if e.channel_id == channel_id]

        # Return newest first, limited
        return list(reversed(events))[:limit]

    def get_counts(self) -> Dict[str, Any]:
        """Get event count statistics."""
        with self._lock:
            return {
                'total_events': len(self._events),
                'by_type': {k.value: v for k, v in self._event_counts.items()},
                'by_severity': {k.name: v for k, v in self._severity_counts.items()},
                'session_start_us': self._session_start_us,
                'batches_processed': self._batch_counter,
            }

    def get_recent(self, n: int = 10) -> List[AuditEvent]:
        """Get N most recent events."""
        with self._lock:
            return list(self._events)[-n:]

    def clear(self) -> None:
        """Clear all logged events."""
        with self._lock:
            self._events.clear()
            self._event_counts.clear()
            self._severity_counts.clear()
