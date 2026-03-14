"""Panel builders for the Rich dashboard."""

from collections.abc import Sequence
from dataclasses import dataclass

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..core.constants import ISI_VIOLATION_LIMIT, MAX_EVENT_LOG_SIZE, REGION_DEFINITIONS, SNR_THRESHOLD
from ..core.data_types import ChannelMetrics


@dataclass
class DashboardEvent:
    """Event for display in the event log."""

    timestamp: str
    event_type: str
    message: str
    severity: str = "INFO"


def build_region_table(metrics: ChannelMetrics | None) -> Table:
    """Create a table showing viability by brain region."""

    table = Table(
        title="Region Viability",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Region", style="cyan", width=20)
    table.add_column("Viable", justify="right", width=10)
    table.add_column("Total", justify="right", width=10)
    table.add_column("Viability %", justify="right", width=12)
    table.add_column("Status", justify="center", width=10)

    if metrics is None:
        table.add_row("Waiting for data...", "", "", "", "")
        return table

    for region_name, (start, end) in REGION_DEFINITIONS.items():
        viable, total, pct = metrics.get_region_viability(start, end)

        if pct >= 0.8:
            status = Text("[OK]", style="bold green")
            pct_style = "green"
        elif pct >= 0.5:
            status = Text("[WARN]", style="bold yellow")
            pct_style = "yellow"
        else:
            status = Text("[FAIL]", style="bold red")
            pct_style = "red"

        pct_text = Text(f"{pct * 100:.1f}%")
        pct_text.stylize(pct_style)
        table.add_row(region_name, str(viable), str(total), pct_text, status)

    return table


def build_stats_panel(
    metrics: ChannelMetrics | None,
    latencies: Sequence[float],
) -> Panel:
    """Create the system statistics panel."""

    if metrics is None:
        return Panel("Waiting for data...", title="System Statistics")

    total_viable = metrics.viable_channel_count
    total_channels = len(metrics.viability_mask)
    overall_pct = total_viable / total_channels * 100
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    stats_text = Text()
    stats_text.append(f"Viable Channels: {total_viable}/{total_channels} ", style="bold")
    stats_text.append(
        f"({overall_pct:.1f}%)\n",
        style="green" if overall_pct >= 80 else "yellow",
    )
    stats_text.append(f"Mean SNR: {float(metrics.snr.mean()):.2f} ")
    stats_text.append(f"(threshold: {SNR_THRESHOLD})\n", style="dim")
    stats_text.append(
        f"Mean ISI Violation: {float(metrics.isi_violation_rate.mean()) * 100:.2f}% "
    )
    stats_text.append(f"(limit: {ISI_VIOLATION_LIMIT * 100}%)\n", style="dim")
    stats_text.append(f"Mean Firing Rate: {float(metrics.firing_rate_hz.mean()):.1f} Hz\n")
    stats_text.append(f"Avg Batch Latency: {avg_latency:.1f} ms", style="dim")

    return Panel(stats_text, title="System Statistics", box=box.ROUNDED)


def build_event_log_panel(events: Sequence[DashboardEvent]) -> Panel:
    """Create the recent-events panel."""

    log_text = Text()
    if not events:
        log_text.append("No events yet...", style="dim")
    else:
        for event in list(events)[:MAX_EVENT_LOG_SIZE]:
            if event.severity == "ERROR":
                style = "bold red"
            elif event.severity == "WARNING":
                style = "yellow"
            else:
                style = "dim"

            log_text.append(f"[{event.timestamp}] ", style="dim")
            log_text.append(f"{event.event_type}: ", style=style)
            log_text.append(f"{event.message}\n")

    return Panel(log_text, title="Event Log", box=box.ROUNDED)