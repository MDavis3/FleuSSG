from types import SimpleNamespace

import numpy as np
import pytest

from ssg import cli as cli_module
from ssg import main as main_module
from ssg.audit.event_types import EventType
from ssg.bench.test_harness import PerformanceValidationResult
from ssg.core.data_types import ChannelMetrics


def test_main_runs_validation_branch(monkeypatch):
    class DummyHarness:
        def __init__(self, seed=None):
            self.seed = seed

        def validate_performance(self, target_latency_ms, duration_sec):
            return PerformanceValidationResult(
                passed=True,
                avg_latency_ms=1.0,
                max_latency_ms=2.0,
                target_latency_ms=target_latency_ms,
                total_batches=5,
                batches_per_second=10.0,
            )

    monkeypatch.setattr(main_module, "TestHarness", DummyHarness)

    assert main_module.main(["test", "--validate-performance", "--duration", "0.1"]) == 0


def test_main_runs_gateway_branch(monkeypatch):
    class DummyGateway:
        created = None

        def __init__(self, config):
            self.config = config
            self.ran = False
            self.exported_to = None
            DummyGateway.created = self

        def run(self, duration_sec, headless):
            self.ran = (duration_sec, headless)

        def export_logs(self, output_path):
            self.exported_to = output_path

    monkeypatch.setattr(main_module, "SignalStabilityGateway", DummyGateway)

    result = main_module.main(
        ["run", "--duration", "1.5", "--headless", "--export-logs", "audit.json"]
    )

    assert result == 0
    assert DummyGateway.created is not None
    assert DummyGateway.created.config.seed is None
    assert DummyGateway.created.ran == (1.5, True)
    assert DummyGateway.created.exported_to == "audit.json"


def test_main_runs_snapshot_branch(monkeypatch, capsys):
    class DummySnapshot:
        def to_text(self):
            return "snapshot output"

    def fake_capture_dashboard_snapshot(*, seed, batches, width):
        assert seed == 7
        assert batches == 3
        assert width == 120
        return DummySnapshot()

    monkeypatch.setattr(
        main_module,
        "capture_dashboard_snapshot",
        fake_capture_dashboard_snapshot,
    )

    assert main_module.main(
        ["snapshot", "--seed", "7", "--batches", "3", "--width", "120"]
    ) == 0
    assert capsys.readouterr().out == "snapshot output\n"


def test_gateway_process_batch_surfaces_latency_warning_in_dashboard(monkeypatch):
    metrics = ChannelMetrics(
        timestamp_us=0,
        snr=np.array([5.0, 4.0], dtype=np.float32),
        firing_rate_hz=np.array([10.0, 9.0], dtype=np.float32),
        isi_violation_rate=np.zeros(2, dtype=np.float32),
        impedance_kohm=np.array([500.0, 500.0], dtype=np.float32),
        viability_mask=np.array([True, False], dtype=bool),
        viable_channel_count=1,
    )
    batch = SimpleNamespace(
        metrics=metrics,
        sanitized_frame=SimpleNamespace(
            artifact_flags=np.zeros(2, dtype=bool),
        ),
    )

    class FakeRuntime:
        def __init__(self):
            self.config = SimpleNamespace(n_channels=2, sample_rate_hz=1000)

        def process_next_batch(self):
            return batch

    class FakeDashboard:
        def __init__(self):
            self.events = []

        def add_event(self, event_type, message, severity):
            self.events.append((event_type, message, severity))

    class FakeLogger:
        def __init__(self):
            self.logs = []

        def log(self, event_type, message, **kwargs):
            self.logs.append((event_type, message, kwargs))

        def log_artifact(self, *args, **kwargs):
            raise AssertionError("artifact logging should not be triggered")

    perf_samples = iter([10.0, 10.02])
    monkeypatch.setattr(main_module.time, "perf_counter", lambda: next(perf_samples))

    dashboard = FakeDashboard()
    logger = FakeLogger()
    gateway = main_module.SignalStabilityGateway(
        config=main_module.GatewayConfig(n_channels=2, sample_rate_hz=1000),
        dependencies=main_module.GatewayDependencies(
            runtime=FakeRuntime(),
            dashboard=dashboard,
            logger=logger,
        ),
    )

    returned_metrics, latency_ms = gateway._process_batch()

    assert returned_metrics is metrics
    assert latency_ms == pytest.approx(20.0)
    assert dashboard.events == [
        ("LATENCY", "Batch 0 latency 20.0ms exceeds 5ms target", "WARNING")
    ]
    assert logger.logs[0][0] is EventType.DATA_LATENCY_WARNING
    assert logger.logs[0][1] == "Batch 0 latency 20.0ms exceeds 5ms target"


def test_cli_module_reexports_main_entrypoint():
    assert cli_module.main is main_module.main
