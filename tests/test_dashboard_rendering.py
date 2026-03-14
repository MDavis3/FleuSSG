import numpy as np
from rich.console import Console

from ssg.core.data_types import ChannelMetrics
from ssg.dashboard.layout import build_header_panel, build_layout
from ssg.dashboard.rendering import DashboardEvent, build_event_log_panel, build_region_table, build_stats_panel


def _make_metrics() -> ChannelMetrics:
    viability_mask = np.zeros(1024, dtype=bool)
    viability_mask[:171] = True
    viability_mask[171:341] = True
    viability_mask[341:512] = False
    viability_mask[512:682] = True
    viability_mask[682:853] = False
    viability_mask[853:] = True
    return ChannelMetrics(
        timestamp_us=0,
        snr=np.full(1024, 5.0, dtype=np.float32),
        firing_rate_hz=np.full(1024, 12.0, dtype=np.float32),
        isi_violation_rate=np.zeros(1024, dtype=np.float32),
        impedance_kohm=np.full(1024, 1000.0, dtype=np.float32),
        viability_mask=viability_mask,
        viable_channel_count=int(viability_mask.sum()),
    )


def _render_text(renderable) -> str:
    console = Console(record=True, width=120)
    console.print(renderable)
    return console.export_text()


def test_dashboard_rendering_helpers_cover_waiting_and_active_states():
    waiting_table = build_region_table(None)
    waiting_stats = build_stats_panel(None, [])
    active_log = build_event_log_panel(
        [DashboardEvent(timestamp="12:00:00", event_type="SYSTEM", message="ok")]
    )
    layout = build_layout(
        start_time=0.0,
        batch_count=3,
        metrics=_make_metrics(),
        latencies=[1.0, 2.0],
        events=[],
    )

    assert waiting_table.row_count == 1
    assert "Waiting for data..." in str(waiting_stats.renderable)
    assert "SYSTEM" in active_log.renderable.plain
    assert layout["regions"].renderable is not None
    assert layout["stats"].renderable is not None
    assert layout["footer"].renderable is not None


def test_dashboard_rendering_includes_header_statuses_and_event_severity():
    header = build_header_panel(start_time=0.0, batch_count=7)
    region_table = build_region_table(_make_metrics())
    event_log = build_event_log_panel(
        [
            DashboardEvent("12:00:00", "SYSTEM", "booted", "INFO"),
            DashboardEvent("12:00:01", "ARTIFACT", "detected", "WARNING"),
            DashboardEvent("12:00:02", "SYSTEM", "failed", "ERROR"),
        ]
    )

    header_text = _render_text(header)
    region_text = _render_text(region_table)
    event_text = _render_text(event_log)

    assert "Signal Stability Gateway" in header_text
    assert "Batches: 7" in header_text
    assert "[OK]" in region_text
    assert "[FAIL]" in region_text
    assert "SYSTEM: booted" in event_text
    assert "ARTIFACT: detected" in event_text
    assert "SYSTEM: failed" in event_text


def test_dashboard_stats_panel_reports_thresholds_and_latency_average():
    panel = build_stats_panel(_make_metrics(), [1.0, 2.0, 3.0])
    panel_text = _render_text(panel)

    assert "Viable Channels:" in panel_text
    assert "Mean SNR:" in panel_text
    assert "threshold: 4.0" in panel_text
    assert "limit: 3.0%" in panel_text
    assert "Avg Batch Latency: 2.0 ms" in panel_text