import threading

import numpy as np
from rich.console import Console

from ssg.core.constants import N_CHANNELS
from ssg.core.data_types import ChannelMetrics
from ssg.dashboard import cli as dashboard_module
from ssg.dashboard.cli import Dashboard


def _make_metrics() -> ChannelMetrics:
    viability_mask = np.zeros(N_CHANNELS, dtype=bool)
    viability_mask[:512] = True
    return ChannelMetrics(
        timestamp_us=0,
        snr=np.full(N_CHANNELS, 5.0, dtype=np.float32),
        firing_rate_hz=np.full(N_CHANNELS, 12.0, dtype=np.float32),
        isi_violation_rate=np.zeros(N_CHANNELS, dtype=np.float32),
        impedance_kohm=np.full(N_CHANNELS, 1000.0, dtype=np.float32),
        viability_mask=viability_mask,
        viable_channel_count=int(viability_mask.sum()),
    )


def test_dashboard_records_events_and_renders_stats():
    dashboard = Dashboard()
    dashboard.add_event("SYSTEM", "booted", "INFO")
    dashboard.update(_make_metrics(), latency_ms=2.5)

    panel = dashboard._create_stats_panel()

    assert dashboard._event_log[0].message == "booted"
    assert "Viable Channels" in panel.renderable.plain


def test_dashboard_creates_region_and_event_views():
    dashboard = Dashboard()
    dashboard.update(_make_metrics(), latency_ms=1.5)
    dashboard.add_event("ARTIFACT", "detected", "WARNING")

    region_table = dashboard._create_region_table()
    event_panel = dashboard._create_event_log()

    assert region_table.row_count == 6
    assert "ARTIFACT" in event_panel.renderable.plain


def test_dashboard_waiting_views_are_explicit():
    dashboard = Dashboard()

    region_table = dashboard._create_region_table()
    stats_panel = dashboard._create_stats_panel()
    event_panel = dashboard._create_event_log()

    assert region_table.row_count == 1
    assert "Waiting for data..." in str(region_table.columns[0]._cells[0])
    assert "Waiting for data..." in str(stats_panel.renderable)
    assert "No events yet..." in event_panel.renderable.plain


def test_dashboard_run_live_updates_and_prints_summary(monkeypatch):
    updated_layouts = []

    class FakeLive:
        def __init__(self, layout, console, refresh_per_second, screen):
            updated_layouts.append(layout)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def update(self, layout):
            updated_layouts.append(layout)

    dashboard = Dashboard()
    dashboard.console = Console(record=True, width=120)
    stop_event = threading.Event()

    def callback():
        stop_event.set()
        return _make_metrics(), 1.0

    monkeypatch.setattr(dashboard_module, "Live", FakeLive)
    monkeypatch.setattr(dashboard_module.time, "sleep", lambda _: None)

    dashboard.run_live(update_callback=callback, stop_event=stop_event)
    dashboard.print_summary()

    assert dashboard._batch_count == 1
    assert updated_layouts
    assert "Session Summary" in dashboard.console.export_text()
