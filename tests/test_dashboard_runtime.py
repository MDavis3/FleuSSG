import threading

from rich.console import Console

from ssg.dashboard.runtime import build_session_summary, run_dashboard_loop


class FakeLive:
    def __init__(self, layout, console, refresh_per_second, screen):
        self.layouts = [layout]
        self.console = console
        self.refresh_per_second = refresh_per_second
        self.screen = screen

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def update(self, layout):
        self.layouts.append(layout)


def test_dashboard_runtime_builds_summary_and_runs_single_iteration():
    console = Console(record=True, width=120)
    stop_event = threading.Event()
    updates = []
    created_layouts = []
    fake_live_holder = {}

    def create_layout():
        layout = {"count": len(created_layouts)}
        created_layouts.append(layout)
        return layout

    def apply_update(metrics, latency):
        updates.append((metrics, latency))

    def callback():
        stop_event.set()
        return ("metrics", 2.5)

    def live_factory(*args, **kwargs):
        fake_live_holder["instance"] = FakeLive(*args, **kwargs)
        return fake_live_holder["instance"]

    run_dashboard_loop(
        console=console,
        create_layout=create_layout,
        refresh_hz=2.0,
        apply_update=apply_update,
        update_callback=callback,
        stop_event=stop_event,
        live_cls=live_factory,
        sleep_fn=lambda _: None,
    )
    summary = build_session_summary(batch_count=3, start_time=0.0)
    summary_console = Console(record=True, width=120)
    summary_console.print(summary)
    summary_output = summary_console.export_text()

    assert updates == [("metrics", 2.5)]
    assert len(created_layouts) >= 2
    assert fake_live_holder["instance"].refresh_per_second == 2.0
    assert fake_live_holder["instance"].screen is True
    assert "Session Summary" in summary_output
    assert "Total Batches: 3" in summary_output