import numpy as np

from ssg.core.data_types import ChannelMetrics
from ssg.dashboard.cli import Dashboard


def test_dashboard_cli_module_renders_summary_text():
    viability_mask = np.array([True, True, False, False], dtype=bool)
    metrics = ChannelMetrics(
        timestamp_us=0,
        snr=np.full(4, 5.0, dtype=np.float32),
        firing_rate_hz=np.full(4, 12.0, dtype=np.float32),
        isi_violation_rate=np.zeros(4, dtype=np.float32),
        impedance_kohm=np.full(4, 1000.0, dtype=np.float32),
        viability_mask=viability_mask,
        viable_channel_count=2,
    )

    dashboard = Dashboard()
    dashboard.update(metrics, latency_ms=2.0)

    panel = dashboard._create_stats_panel()

    assert "Viable Channels" in panel.renderable.plain
