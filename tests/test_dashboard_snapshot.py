from types import SimpleNamespace

import numpy as np

from ssg.core.data_types import ChannelMetrics, SanitizedFrame
from ssg.dashboard import snapshot as snapshot_module


def test_capture_dashboard_snapshot_collects_rendered_metrics(monkeypatch):
    metrics = ChannelMetrics(
        timestamp_us=123,
        snr=np.array([5.0, 4.0], dtype=np.float32),
        firing_rate_hz=np.array([10.0, 8.0], dtype=np.float32),
        isi_violation_rate=np.array([0.0, 0.01], dtype=np.float32),
        impedance_kohm=np.array([500.0, 550.0], dtype=np.float32),
        viability_mask=np.array([True, False], dtype=bool),
        viable_channel_count=1,
    )
    sanitized = SanitizedFrame(
        timestamp_us=123,
        raw_unfiltered=np.zeros((4, 2), dtype=np.float32),
        lfp=np.zeros((4, 2), dtype=np.float32),
        spikes=np.zeros((4, 2), dtype=np.float32),
        artifact_flags=np.zeros(2, dtype=bool),
    )

    class FakeRuntime:
        def __init__(self, config):
            self.config = SimpleNamespace(
                n_channels=2,
                sample_rate_hz=1000,
                batch_size=100,
            )

        def process_next_batch(self):
            return SimpleNamespace(metrics=metrics, sanitized_frame=sanitized)

    class FakeDashboard:
        def __init__(self):
            self._event_log = []

        def add_event(self, event_type, message, severity="INFO"):
            self._event_log.insert(
                0,
                SimpleNamespace(
                    timestamp="12:00:00",
                    event_type=event_type,
                    message=message,
                    severity=severity,
                ),
            )

        def update(self, *_args, **_kwargs):
            return None

        def _create_layout(self):
            return "fake layout"

    class FakeConsole:
        def __init__(self, record=True, width=160):
            self.rendered = []

        def print(self, value):
            self.rendered.append(str(value))

        def export_text(self):
            return "\n".join(self.rendered)

    perf_samples = iter([0.0, 0.02, 0.03, 0.05])
    monkeypatch.setattr(snapshot_module, "PipelineRuntime", FakeRuntime)
    monkeypatch.setattr(snapshot_module, "Dashboard", FakeDashboard)
    monkeypatch.setattr(snapshot_module, "Console", FakeConsole)
    monkeypatch.setattr(snapshot_module, "REGION_DEFINITIONS", {"Test Region": (0, 2)})
    monkeypatch.setattr(
        snapshot_module.time,
        "perf_counter",
        lambda: next(perf_samples),
    )

    snapshot = snapshot_module.capture_dashboard_snapshot(seed=7, batches=2, width=120)

    assert snapshot.rendered_text == "fake layout"
    assert snapshot.seed == 7
    assert snapshot.batches_rendered == 2
    assert snapshot.batch_duration_ms == 100.0
    assert snapshot.viable_channels == 1
    assert snapshot.avg_latency_ms == 20.0
    assert snapshot.regions[0].name == "Test Region"
    assert any(
        "LATENCY: Batch 1 latency 20.0ms exceeds 5ms target" in event
        for event in snapshot.event_log
    )
    assert "=== NUMERIC SPECS ===" in snapshot.to_text()
