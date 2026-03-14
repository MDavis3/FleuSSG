from ssg import main as main_module
from ssg.bench.test_harness import PerformanceValidationResult


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
