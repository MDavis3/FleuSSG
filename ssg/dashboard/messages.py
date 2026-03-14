"""Shared dashboard-facing message helpers."""

from ..core.constants import LATENCY_WARNING_THRESHOLD_MS


def format_artifact_event_message(affected_channels: int) -> str:
    """Format the artifact event shown in the dashboard log."""

    return f"{affected_channels} channels affected"


def format_latency_warning_message(
    batch_id: int,
    latency_ms: float,
    target_ms: float = LATENCY_WARNING_THRESHOLD_MS,
) -> str:
    """Format a stable latency warning for logs and the dashboard."""

    return (
        f"Batch {batch_id} latency {latency_ms:.1f}ms exceeds "
        f"{target_ms:.0f}ms target"
    )
