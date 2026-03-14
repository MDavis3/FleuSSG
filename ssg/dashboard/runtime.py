"""Runtime helpers for the CLI dashboard."""

import time
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel


LayoutFactory = Callable[[], Any]
ApplyUpdate = Callable[[Any, float], None]
UpdateCallback = Callable[[], tuple[Any, float] | None]


def build_session_summary(batch_count: int, start_time: float) -> Panel:
    """Build the dashboard session summary panel."""

    return Panel(
        f"Session Summary\n"
        f"Total Batches: {batch_count}\n"
        f"Runtime: {time.time() - start_time:.1f}s",
        title="SSG Dashboard Closed",
        border_style="cyan",
    )


def run_dashboard_loop(
    *,
    console: Console,
    create_layout: LayoutFactory,
    refresh_hz: float,
    apply_update: ApplyUpdate,
    update_callback: UpdateCallback = None,
    stop_event=None,
    live_cls=Live,
    sleep_fn=time.sleep,
) -> None:
    """Run the dashboard refresh loop."""

    refresh_interval = 1.0 / refresh_hz

    with live_cls(
        create_layout(),
        console=console,
        refresh_per_second=refresh_hz,
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
                        apply_update(metrics, latency)

                live.update(create_layout())
                sleep_fn(refresh_interval)
        except KeyboardInterrupt:
            if stop_event:
                stop_event.set()
            raise
