"""
FDA-Compliant Event Types for Signal Stability Gateway

Enumerated event types for regulatory traceability.
All events are timestamped and can be exported for audit.
"""

from enum import Enum, auto


class EventSeverity(Enum):
    """Severity levels for audit events."""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class EventType(Enum):
    """
    Enumerated event types for FDA audit trail.

    Categories:
        SYSTEM_*: System lifecycle events
        CHANNEL_*: Per-channel status changes
        ARTIFACT_*: Artifact detection events
        VALIDATION_*: Validation threshold events
        DATA_*: Data flow events
    """
    # System Events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    SYSTEM_ERROR = "system_error"
    SYSTEM_CONFIG_CHANGE = "system_config_change"

    # Channel Events
    CHANNEL_VIABLE = "channel_viable"
    CHANNEL_NON_VIABLE = "channel_non_viable"
    CHANNEL_SNR_LOW = "channel_snr_low"
    CHANNEL_SNR_RECOVERED = "channel_snr_recovered"
    CHANNEL_ISI_VIOLATION = "channel_isi_violation"
    CHANNEL_IMPEDANCE_OUT_OF_RANGE = "channel_impedance_oor"

    # Artifact Events
    ARTIFACT_DETECTED = "artifact_detected"
    ARTIFACT_JAW_CLENCH = "artifact_jaw_clench"
    ARTIFACT_ELECTRODE_DRIFT = "artifact_electrode_drift"
    ARTIFACT_MOTION_SPIKE = "artifact_motion_spike"

    # Validation Events
    VALIDATION_THRESHOLD_BREACH = "validation_threshold_breach"
    VALIDATION_VIABILITY_DROP = "validation_viability_drop"
    VALIDATION_VIABILITY_RECOVERED = "validation_viability_recovered"

    # Data Events
    DATA_BATCH_PROCESSED = "data_batch_processed"
    DATA_BUFFER_OVERFLOW = "data_buffer_overflow"
    DATA_LATENCY_WARNING = "data_latency_warning"


# Event descriptions for documentation
EVENT_DESCRIPTIONS = {
    EventType.SYSTEM_START: "SSG system initialized and started",
    EventType.SYSTEM_STOP: "SSG system stopped gracefully",
    EventType.SYSTEM_ERROR: "System-level error occurred",
    EventType.SYSTEM_CONFIG_CHANGE: "Configuration parameter changed",

    EventType.CHANNEL_VIABLE: "Channel became viable (meets all thresholds)",
    EventType.CHANNEL_NON_VIABLE: "Channel became non-viable",
    EventType.CHANNEL_SNR_LOW: "Channel SNR dropped below threshold",
    EventType.CHANNEL_SNR_RECOVERED: "Channel SNR recovered above threshold",
    EventType.CHANNEL_ISI_VIOLATION: "Channel ISI violation rate exceeded limit",
    EventType.CHANNEL_IMPEDANCE_OUT_OF_RANGE: "Channel impedance outside valid range",

    EventType.ARTIFACT_DETECTED: "General artifact detected",
    EventType.ARTIFACT_JAW_CLENCH: "Jaw clench artifact (EMG burst) detected",
    EventType.ARTIFACT_ELECTRODE_DRIFT: "Electrode drift (DC shift) detected",
    EventType.ARTIFACT_MOTION_SPIKE: "Motion spike (transient) detected",

    EventType.VALIDATION_THRESHOLD_BREACH: "Validation threshold breached",
    EventType.VALIDATION_VIABILITY_DROP: "Overall viability dropped significantly",
    EventType.VALIDATION_VIABILITY_RECOVERED: "Overall viability recovered",

    EventType.DATA_BATCH_PROCESSED: "Data batch processed successfully",
    EventType.DATA_BUFFER_OVERFLOW: "Ring buffer overflow occurred",
    EventType.DATA_LATENCY_WARNING: "Processing latency exceeded target",
}
