"""Rich CLI dashboard for Signal Stability Gateway."""

import time
from collections import deque
from typing import Deque, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from ..core.constants import DASHBOARD_REFRESH_HZ, MAX_EVENT_LOG_SIZE
from ..core.data_types import ChannelMetrics
from .layout import build_layout
from .rendering import DashboardEvent, build_event_log_panel, build_region_table, build_stats_panel
from .runtime import build_session_summary, run_dashboard_loop


class Dashboard:
    """Real-time CLI dashboard using Rich."""

    def __init__(self):
        self.console = Console()
        self._latest_metrics: Optional[ChannelMetrics] = None
        self._event_log: Deque[DashboardEvent] = deque(maxlen=MAX_EVENT_LOG_SIZE)
        self._start_time = time.time()
        self._batch_count = 0
        self._last_update_time = time.time()
        self._latencies: Deque[float] = deque(maxlen=100)

    def update(self, metrics: ChannelMetrics, latency_ms: float = 0.0) -> None:
        """Update dashboard with new metrics."""

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

        self._event_log.appendleft(
            DashboardEvent(
                timestamp=time.strftime("%H:%M:%S"),
                event_type=event_type,
                message=message,
                severity=severity,
            )
        )

    def _create_region_table(self):
        """Create table showing viability by brain region."""

        return build_region_table(self._latest_metrics)

    def _create_stats_panel(self) -> Panel:
        """Create system-wide statistics panel."""

        return build_stats_panel(self._latest_metrics, self._latencies)

    def _create_event_log(self) -> Panel:
        """Create recent events panel."""

        return build_event_log_panel(list(self._event_log)[:MAX_EVENT_LOG_SIZE])

    def _create_layout(self):
        """Create full dashboard layout."""

        return build_layout(
            start_time=self._start_time,
            batch_count=self._batch_count,
            metrics=self._latest_metrics,
            latencies=list(self._latencies),
            events=list(self._event_log)[:MAX_EVENT_LOG_SIZE],
        )

    def render_once(self) -> None:
        """Render dashboard once (for testing)."""

        self.console.clear()
        self.console.print(self._create_layout())

    def run_live(
        self,
        update_callback=None,
        stop_event=None,
    ) -> None:
        """Run the live dashboard with automatic refresh."""

        run_dashboard_loop(
            console=self.console,
            create_layout=self._create_layout,
            refresh_hz=DASHBOARD_REFRESH_HZ,
            apply_update=self.update,
            update_callback=update_callback,
            stop_event=stop_event,
            live_cls=Live,
            sleep_fn=time.sleep,
        )

    def print_summary(self) -> None:
        """Print final summary when dashboard closes."""

        self.console.print("\n")
        self.console.print(build_session_summary(self._batch_count, self._start_time))