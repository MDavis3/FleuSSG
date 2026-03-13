"""
Rich CLI Dashboard for Signal Stability Gateway

Real-time visualization of channel viability and system metrics.
Refreshes at 2Hz for responsive monitoring without CPU overhead.
"""

import time
from typing import Optional, List, Deque
from collections import deque
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn
from rich import box

from ..core.constants import (
    REGION_DEFINITIONS,
    DASHBOARD_REFRESH_HZ,
    MAX_EVENT_LOG_SIZE,
    SNR_THRESHOLD,
    ISI_VIOLATION_LIMIT,
)
from ..core.data_types import ChannelMetrics


@dataclass
class DashboardEvent:
    """Event for display in the event log."""
    timestamp: str
    event_type: str
    message: str
    severity: str = "INFO"


class Dashboard:
    """
    Real-time CLI dashboard using Rich library.

    Displays:
        - Region viability overview (6 brain regions)
        - System-wide statistics
        - Recent event log
        - Performance metrics

    Refresh Rate: 2Hz (configurable via DASHBOARD_REFRESH_HZ)
    """

    def __init__(self):
        """Initialize dashboard components."""
        self.console = Console()
        self._latest_metrics: Optional[ChannelMetrics] = None
        self._event_log: Deque[DashboardEvent] = deque(maxlen=MAX_EVENT_LOG_SIZE)
        self._start_time = time.time()
        self._batch_count = 0
        self._last_update_time = time.time()

        # Performance tracking
        self._latencies: Deque[float] = deque(maxlen=100)

    def update(self, metrics: ChannelMetrics, latency_ms: float = 0.0) -> None:
        """
        Update dashboard with new metrics.

        Args:
            metrics: Latest ChannelMetrics from ValidationEngine
            latency_ms: Processing latency for this batch
        """
        self._latest_metrics = metrics
        self._batch_count += 1
        self._last_update_time = time.time()

        if latency_ms > 0:
            self._latencies.append(latency_ms)

    def add_event(
        self,
        event_type: str,
        message: str,
        severity: str = "INFO",
    ) -> None:
        """Add event to the log."""
        timestamp = time.strftime("%H:%M:%S")
        event = DashboardEvent(
            timestamp=timestamp,
            event_type=event_type,
            message=message,
            severity=severity,
        )
        self._event_log.appendleft(event)

    def _create_header(self) -> Panel:
        """Create dashboard header panel."""
        elapsed = time.time() - self._start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)

        header_text = Text()
        header_text.append("Signal Stability Gateway", style="bold cyan")
        header_text.append(" | ")
        header_text.append(f"Runtime: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}", style="dim")
        header_text.append(" | ")
        header_text.append(f"Batches: {self._batch_count}", style="dim")

        return Panel(header_text, box=box.ROUNDED)

    def _create_region_table(self) -> Table:
        """Create table showing viability by brain region."""
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

        if self._latest_metrics is None:
            table.add_row("Waiting for data...", "", "", "", "")
            return table

        for region_name, (start, end) in REGION_DEFINITIONS.items():
            viable, total, pct = self._latest_metrics.get_region_viability(start, end)

            # Status indicator
            if pct >= 0.8:
                status = Text("[OK]", style="bold green")
            elif pct >= 0.5:
                status = Text("[WARN]", style="bold yellow")
            else:
                status = Text("[FAIL]", style="bold red")

            # Viability percentage with color
            pct_text = Text(f"{pct * 100:.1f}%")
            if pct >= 0.8:
                pct_text.stylize("green")
            elif pct >= 0.5:
                pct_text.stylize("yellow")
            else:
                pct_text.stylize("red")

            table.add_row(
                region_name,
                str(viable),
                str(total),
                pct_text,
                status,
            )

        return table

    def _create_stats_panel(self) -> Panel:
        """Create system-wide statistics panel."""
        if self._latest_metrics is None:
            return Panel("Waiting for data...", title="System Statistics")

        total_viable = self._latest_metrics.viable_channel_count
        total_channels = len(self._latest_metrics.viability_mask)
        overall_pct = total_viable / total_channels * 100

        # Compute average metrics
        mean_snr = float(self._latest_metrics.snr.mean())
        mean_isi = float(self._latest_metrics.isi_violation_rate.mean()) * 100
        mean_fr = float(self._latest_metrics.firing_rate_hz.mean())

        # Latency stats
        avg_latency = sum(self._latencies) / len(self._latencies) if self._latencies else 0

        stats_text = Text()
        stats_text.append(f"Viable Channels: {total_viable}/{total_channels} ", style="bold")
        stats_text.append(f"({overall_pct:.1f}%)\n", style="green" if overall_pct >= 80 else "yellow")

        stats_text.append(f"Mean SNR: {mean_snr:.2f} ")
        stats_text.append(f"(threshold: {SNR_THRESHOLD})\n", style="dim")

        stats_text.append(f"Mean ISI Violation: {mean_isi:.2f}% ")
        stats_text.append(f"(limit: {ISI_VIOLATION_LIMIT * 100}%)\n", style="dim")

        stats_text.append(f"Mean Firing Rate: {mean_fr:.1f} Hz\n")

        stats_text.append(f"Avg Batch Latency: {avg_latency:.1f} ms", style="dim")

        return Panel(stats_text, title="System Statistics", box=box.ROUNDED)

    def _create_event_log(self) -> Panel:
        """Create recent events panel."""
        log_text = Text()

        if not self._event_log:
            log_text.append("No events yet...", style="dim")
        else:
            for event in list(self._event_log)[:MAX_EVENT_LOG_SIZE]:
                # Severity color
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

    def _create_layout(self) -> Layout:
        """Create full dashboard layout."""
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

        # Populate layout
        layout["header"].update(self._create_header())
        layout["regions"].update(self._create_region_table())
        layout["stats"].update(self._create_stats_panel())
        layout["footer"].update(self._create_event_log())

        return layout

    def render_once(self) -> None:
        """Render dashboard once (for testing)."""
        self.console.clear()
        self.console.print(self._create_layout())

    def run_live(
        self,
        update_callback=None,
        stop_event=None,
    ) -> None:
        """
        Run live dashboard with automatic refresh.

        Args:
            update_callback: Optional callback that returns (metrics, latency_ms)
            stop_event: Optional threading.Event to signal stop
        """
        refresh_interval = 1.0 / DASHBOARD_REFRESH_HZ

        with Live(
            self._create_layout(),
            console=self.console,
            refresh_per_second=DASHBOARD_REFRESH_HZ,
            screen=True,
        ) as live:
            try:
                while True:
                    if stop_event and stop_event.is_set():
                        break

                    if update_callback:
                        result = update_callback()
                        if result:
                            metrics, latency = result
                            self.update(metrics, latency)

                    live.update(self._create_layout())
                    time.sleep(refresh_interval)

            except KeyboardInterrupt:
                pass

    def print_summary(self) -> None:
        """Print final summary when dashboard closes."""
        self.console.print("\n")
        self.console.print(Panel(
            f"Session Summary\n"
            f"Total Batches: {self._batch_count}\n"
            f"Runtime: {time.time() - self._start_time:.1f}s",
            title="SSG Dashboard Closed",
            border_style="cyan",
        ))
