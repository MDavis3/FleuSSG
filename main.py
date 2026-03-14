"""CLI entry point and runtime orchestration for the SSG demo pipeline."""

import argparse
import time
import threading
from dataclasses import dataclass
from typing import Sequence

from .core.constants import (
    N_CHANNELS,
    SAMPLE_RATE_HZ,
    BATCH_SIZE,
    BATCH_DURATION_SEC,
)
from .core.data_types import ChannelMetrics
from .bench.test_harness import TestHarness
from .dashboard.cli import Dashboard
from .ingestion.engine import IngestionEngine
from .ingestion.mock_telemetry import MockTelemetry, MockTelemetryConfig
from .audit.audit_logger import AuditEventContext, AuditLogger
from .audit.event_types import EventSeverity, EventType
from .audit.exporters import JSONExporter
from .sanitization.layer import SanitizationLayer
from .validation.engine import ValidationEngine


@dataclass
class GatewayConfig:
    """Runtime configuration for the SSG pipeline."""

    n_channels: int = N_CHANNELS
    sample_rate_hz: int = SAMPLE_RATE_HZ
    seed: int | None = None


@dataclass
class GatewayDependencies:
    """Optional collaborators for the SSG runtime."""
    telemetry: MockTelemetry | None = None
    ingestion: IngestionEngine | None = None
    sanitization: SanitizationLayer | None = None
    validation: ValidationEngine | None = None
    dashboard: Dashboard | None = None
    logger: AuditLogger | None = None


class SignalStabilityGateway:
    """Coordinate ingestion, sanitization, validation, and presentation."""

    def __init__(
        self,
        config: GatewayConfig | None = None,
        dependencies: GatewayDependencies | None = None,
    ):
        self.config = config or GatewayConfig()
        self.n_channels = self.config.n_channels
        self.sample_rate = self.config.sample_rate_hz
        self._dependencies = dependencies or GatewayDependencies()

        self._telemetry = self._dependencies.telemetry or MockTelemetry(
            n_channels=self.n_channels,
            sample_rate_hz=self.sample_rate,
            config=MockTelemetryConfig(seed=self.config.seed),
        )
        self._ingestion = self._dependencies.ingestion or IngestionEngine(
            n_channels=self.n_channels,
            sample_rate_hz=self.sample_rate,
        )
        self._sanitization = self._dependencies.sanitization or SanitizationLayer(
            n_channels=self.n_channels,
            sample_rate_hz=self.sample_rate,
        )
        self._validation = self._dependencies.validation or ValidationEngine(
            n_channels=self.n_channels,
            sample_rate_hz=self.sample_rate,
        )
        self._dashboard = self._dependencies.dashboard
        self._logger = self._dependencies.logger or AuditLogger()

        self._running = False
        self._stop_event = threading.Event()
        self._batch_count = 0

    def _get_dashboard(self) -> Dashboard:
        """Create the dashboard on demand when the CLI needs it."""
        if self._dashboard is None:
            self._dashboard = Dashboard()
        return self._dashboard

    def run(
        self,
        duration_sec: float = 60.0,
        headless: bool = False,
    ) -> None:
        """Run the pipeline for a fixed duration or until interrupted."""
        self._running = True
        self._stop_event.clear()
        self._logger.log(
            EventType.SYSTEM_START,
            f"SSG started: {self.n_channels} channels, {self.sample_rate}Hz",
            severity=EventSeverity.INFO,
        )

        start_time = time.time()
        max_batches = (
            int(duration_sec * self.sample_rate / BATCH_SIZE)
            if duration_sec > 0
            else None
        )

        try:
            if headless:
                self._run_headless(max_batches, start_time)
            else:
                self._run_with_dashboard(max_batches)
        finally:
            self._running = False
            self._logger.log(
                EventType.SYSTEM_STOP,
                f"SSG stopped after {self._batch_count} batches",
                severity=EventSeverity.INFO,
            )

    def _run_headless(
        self,
        max_batches: int | None,
        start_time: float,
    ) -> None:
        """Run pipeline without dashboard rendering."""
        while not self._stop_event.is_set():
            if max_batches is not None and self._batch_count >= max_batches:
                break
            batch_start = time.perf_counter()
            metrics, latency = self._process_batch()

            if self._batch_count % 10 == 0:
                elapsed = time.time() - start_time
                print(
                    f"Batch {self._batch_count}: "
                    f"{metrics.viable_channel_count}/{self.n_channels} viable, "
                    f"{latency:.1f}ms latency, "
                    f"{elapsed:.1f}s elapsed"
                )

            elapsed = time.perf_counter() - batch_start
            if elapsed < BATCH_DURATION_SEC:
                time.sleep(BATCH_DURATION_SEC - elapsed)

    def _run_with_dashboard(self, max_batches: int | None) -> None:
        """Run pipeline with live dashboard updates."""
        dashboard = self._get_dashboard()

        def update_callback() -> tuple[ChannelMetrics, float] | None:
            if self._stop_event.is_set():
                self._stop_event.set()
                return None

            if max_batches is not None and self._batch_count >= max_batches:
                self._stop_event.set()
                return None

            return self._process_batch()

        dashboard.add_event(
            "SYSTEM",
            f"Started: {self.n_channels} channels @ {self.sample_rate}Hz",
            "INFO",
        )
        dashboard.run_live(
            update_callback=update_callback,
            stop_event=self._stop_event,
        )
        dashboard.print_summary()

    def _process_batch(self) -> tuple[ChannelMetrics, float]:
        """Process a single telemetry batch end to end."""
        batch_start = time.perf_counter()
        samples, timestamps = self._telemetry.generate_batch(BATCH_SIZE)

        self._ingestion.ingest_batch(samples, timestamps)
        sanitized = self._sanitization.process(samples, timestamps)
        metrics = self._validation.process(sanitized, timestamps)
        latency_ms = (time.perf_counter() - batch_start) * 1000

        if sanitized.artifact_flags.any():
            n_affected = int(sanitized.artifact_flags.sum())
            self._logger.log_artifact(
                EventType.ARTIFACT_DETECTED,
                affected_channels=n_affected,
                amplitude=1.0,
                duration_ms=BATCH_DURATION_SEC * 1000,
                batch_id=self._batch_count,
            )
            if self._dashboard is not None:
                self._dashboard.add_event(
                    "ARTIFACT",
                    f"{n_affected} channels affected",
                    "WARNING",
                )

        if latency_ms > 5.0:
            self._logger.log(
                EventType.DATA_LATENCY_WARNING,
                f"Batch latency {latency_ms:.1f}ms exceeds 5ms target",
                severity=EventSeverity.WARNING,
                context=AuditEventContext(batch_id=self._batch_count),
            )

        self._batch_count += 1
        return metrics, latency_ms

    def stop(self) -> None:
        """Stop the pipeline gracefully."""
        self._stop_event.set()

    def export_logs(self, output_path: str) -> None:
        """Export audit logs to a JSON file."""
        JSONExporter.export_logger(self._logger, output_path)
        print(f"Logs exported to: {output_path}")


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Signal Stability Gateway - Neural data simulation pipeline",
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    run_parser = subparsers.add_parser('run', help='Run the SSG pipeline')
    run_parser.add_argument(
        '--duration', '-d',
        type=float,
        default=60.0,
        help='Duration in seconds (0 for indefinite)',
    )
    run_parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without dashboard rendering',
    )
    run_parser.add_argument(
        '--channels', '-c',
        type=int,
        default=N_CHANNELS,
        help=f'Number of channels (default: {N_CHANNELS})',
    )
    run_parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility',
    )
    run_parser.add_argument(
        '--export-logs',
        type=str,
        default=None,
        help='Export logs to JSON on exit',
    )

    test_parser = subparsers.add_parser('test', help='Run the simulation harness')
    test_parser.add_argument(
        '--duration', '-d',
        type=float,
        default=10.0,
        help='Test duration in seconds',
    )
    test_parser.add_argument(
        '--validate-performance',
        action='store_true',
        help='Run performance validation only',
    )
    test_parser.add_argument(
        '--no-artifacts',
        action='store_true',
        help='Disable artifact injection',
    )
    test_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == 'run':
            gateway = SignalStabilityGateway(
                config=GatewayConfig(
                    n_channels=args.channels,
                    seed=args.seed,
                ),
            )
            gateway.run(
                duration_sec=args.duration,
                headless=args.headless,
            )

            if args.export_logs:
                gateway.export_logs(args.export_logs)
            return 0

        if args.command == 'test':
            harness = TestHarness(seed=args.seed)

            if args.validate_performance:
                print("Running performance validation...")
                results = harness.validate_performance(
                    target_latency_ms=5.0,
                    duration_sec=args.duration,
                )
                print(
                    f"\nPerformance Validation: {'PASSED' if results.passed else 'FAILED'}"
                )
                print(f"  Average Latency: {results.avg_latency_ms:.2f}ms")
                print(f"  Max Latency: {results.max_latency_ms:.2f}ms")
                print(f"  Target: {results.target_latency_ms:.2f}ms")
                print(f"  Batches/sec: {results.batches_per_second:.1f}")
                return 0 if results.passed else 1

            print(f"Running test harness for {args.duration}s...")
            results = harness.run(
                duration_sec=args.duration,
                inject_artifacts=not args.no_artifacts,
            )
            print("\nTest Results:")
            print(f"  Total Batches: {results.total_batches}")
            print(f"  Duration: {results.total_duration_sec:.2f}s")
            print(f"  Avg Latency: {results.avg_latency_ms:.2f}ms")
            print(f"  Max Latency: {results.max_latency_ms:.2f}ms")
            print(
                "  Viable Channels: "
                f"{results.min_viable_channels}-{results.max_viable_channels} "
                f"(avg: {results.avg_viable_channels:.0f})"
            )
            print(f"  Artifacts Injected: {results.total_artifacts_injected}")
            print(
                "  SNR: "
                f"min={results.snr_distribution.min:.2f}, "
                f"max={results.snr_distribution.max:.2f}, "
                f"mean={results.snr_distribution.mean:.2f}"
            )
            return 0

        parser.error(f"Unknown command: {args.command}")
    except KeyboardInterrupt:
        print("\nShutting down...")
        return 130

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
