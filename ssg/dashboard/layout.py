"""Layout assembly helpers for the Rich dashboard."""

import time
from collections.abc import Sequence

from rich import box
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from ..core.data_types import ChannelMetrics
from .rendering import DashboardEvent, build_event_log_panel, build_region_table, build_stats_panel


def build_header_panel(start_time: float, batch_count: int) -> Panel:
    """Create the dashboard header panel."""

    elapsed = time.time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)

    header_text = Text()
    header_text.append("Signal Stability Gateway", style="bold cyan")
    header_text.append(" | ")
    header_text.append(
        f"Runtime: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}",
        style="dim",
    )
    header_text.append(" | ")
    header_text.append(f"Batches: {batch_count}", style="dim")

    return Panel(header_text, box=box.ROUNDED)


def build_layout(
    start_time: float,
    batch_count: int,
    metrics: ChannelMetrics | None,
    latencies: Sequence[float],
    events: Sequence[DashboardEvent],
) -> Layout:
    """Create the full dashboard layout."""

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=12),
    )
    layout["main"].split_row(
        Layout(name="regions", ratio=2),
        Layout(name="stats", ratio=1),
    )
    layout["header"].update(build_header_panel(start_time, batch_count))
    layout["regions"].update(build_region_table(metrics))
    layout["stats"].update(build_stats_panel(metrics, latencies))
    layout["footer"].update(build_event_log_panel(events))
    return layout