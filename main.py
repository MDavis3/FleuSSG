"""
Signal Stability Gateway - Main Entry Point

Mission-critical middleware bridging high-density neural electrode arrays
with AI orchestration and foundation models.

Usage:
    python -m ssg.main --help
    python -m ssg.main run --duration 60
    python -m ssg.main test --validate-performance
"""

import argparse
import sys
import time
import threading
from typing import Optional

from .core.constants import (
    N_CHANNELS,
    SAMPLE_RATE_HZ,
    BATCH_SIZE,
    BATCH_DURATION_SEC,
)
from .ingestion.mock_telemetry import MockTelemetry
from .ingestion.engine import IngestionEngine
from .sanitization.layer import SanitizationLayer
from .validation.engine import ValidationEngine
from .dashboard.cli import Dashboard
from .simulation.test_harness import TestHarness
from .logging.audit_logger import AuditLogger
from .logging.event_types import EventType, EventSeverity
from .logging.exporters import JSONExporter


class SignalStabilityGateway:
    """
    Main orchestrator for the SSG pipeline.

    Coordinates:
        - Data ingestion (MockTelemetry or real hardware)
        - Signal sanitization (filters + artifact rejection)
        - Channel validation (SNR, ISI, viability)
        - Dashboard visualization
        - FDA audit logging
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        sample_rate_hz: int = SAMPLE_RATE_HZ,
        inject_artifacts: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialize SSG pipeline.

        Args:
            n_channels: Number of recording channels
            sample_rate_hz: Sampling rate in Hz
            inject_artifacts: Enable artifact injection for testing
            seed: Random seed for reproducibility
        """
        self.n_channels = n_channels
        self.sample_rate = sample_rate_hz
        self.inject_artifacts = inject_artifacts

        # Initialize components
        self._telemetry = MockTelemetry(
            n_channels=n_channels,
            sample_rate_hz=sample_rate_hz,
            seed=seed,
        )

        self._ingestion = IngestionEngine(
            n_channels=n_channels,
            sample_rate_hz=sample_rate_hz,
        )

        self._sanitization = SanitizationLayer(
            n_channels=n_channels,
            sample_rate_hz=sample_rate_hz,
        )

        self._validation = ValidationEngine(
            n_channels=n_channels,
            sample_rate_hz=sample_rate_hz,
        )

        self._dashboard = Dashboard()
        self._logger = AuditLogger()

        # Runtime state
        self._running = False
        self._stop_event = threading.Event()
        self._batch_count = 0

    def run(
        self,
        duration_sec: float = 60.0,
        headless: bool = False,
    ) -> None:
        """
        Run the SSG pipeline.

        Args:
            duration_sec: Duration to run (0 for indefinite)
            headless: Disable dashboard visualization
        """
        self._running = True
        self._stop_event.clear()

        # Log startup
        self._logger.log(
            EventType.SYSTEM_START,
            f"SSG started: {self.n_channels} channels, {self.sample_rate}Hz",
            severity=EventSeverity.INFO,
        )

        start_time = time.time()
        n_batches = int(duration_sec * self.sample_rate / BATCH_SIZE) if duration_sec > 0 else float('inf')

        try:
            if headless:
                self._run_headless(n_batches, start_time, duration_sec)
            else:
                self._run_with_dashboard(n_batches, start_time, duration_sec)

        except KeyboardInterrupt:
            print("\nShutting down...")

        finally:
            self._running = False
            self._logger.log(
                EventType.SYSTEM_STOP,
                f"SSG stopped after {self._batch_count} batches",
                severity=EventSeverity.INFO,
            )

    def _run_headless(
        self,
        n_batches: float,
        start_time: float,
        duration_sec: float,
    ) -> None:
        """Run pipeline without dashboard."""
        batch_interval = BATCH_DURATION_SEC

        while self._batch_count < n_batches and not self._stop_event.is_set():
            batch_start = time.perf_counter()

            metrics, latency = self._process_batch()

            # Print status every 10 batches
            if self._batch_count % 10 == 0:
                elapsed = time.time() - start_time
                print(
                    f"Batch {self._batch_count}: "
                    f"{metrics.viable_channel_count}/{self.n_channels} viable, "
                    f"{latency:.1f}ms latency, "
                    f"{elapsed:.1f}s elapsed"
                )

            # Rate limiting
            elapsed = time.perf_counter() - batch_start
            if elapsed < batch_interval:
                time.sleep(batch_interval - elapsed)

    def _run_with_dashboard(
        self,
        n_batches: float,
        start_time: float,
        duration_sec: float,
    ) -> None:
        """Run pipeline with Rich dashboard."""

        def update_callback():
            if self._batch_count >= n_batches or self._stop_event.is_set():
                return None

            metrics, latency = self._process_batch()
            return metrics, latency

        self._dashboard.add_event(
            "SYSTEM",
            f"Started: {self.n_channels} channels @ {self.sample_rate}Hz",
            "INFO",
        )

        self._dashboard.run_live(
            update_callback=update_callback,
            stop_event=self._stop_event,
        )

        self._dashboard.print_summary()

    def _process_batch(self):
        """Process a single batch through the pipeline."""
        batch_start = time.perf_counter()

        # Generate synthetic data
        samples, timestamps = self._telemetry.generate_batch(BATCH_SIZE)

        # Ingest
        self._ingestion.ingest_batch(samples, timestamps)

        # Sanitize
        sanitized = self._sanitization.process(samples, timestamps)

        # Validate
        metrics = self._validation.process(sanitized, timestamps)

        latency_ms = (time.perf_counter() - batch_start) * 1000

        # Log artifact events
        if sanitized.artifact_flags.any():
            n_affected = int(sanitized.artifact_flags.sum())
            self._logger.log_artifact(
                EventType.ARTIFACT_DETECTED,
                affected_channels=n_affected,
                amplitude=1.0,
                duration_ms=BATCH_DURATION_SEC * 1000,
                batch_id=self._batch_count,
            )
            self._dashboard.add_event(
                "ARTIFACT",
                f"{n_affected} channels affected",
                "WARNING",
            )

        # Log latency warnings
        if latency_ms > 5.0:
            self._logger.log(
                EventType.DATA_LATENCY_WARNING,
                f"Batch latency {latency_ms:.1f}ms exceeds 5ms target",
                severity=EventSeverity.WARNING,
                batch_id=self._batch_count,
            )

        self._batch_count += 1

        return metrics, latency_ms

    def stop(self) -> None:
        """Stop the pipeline gracefully."""
        self._stop_event.set()

    def export_logs(self, output_path: str) -> None:
        """Export audit logs to JSON file."""
        JSONExporter.export_logger(self._logger, output_path)
        print(f"Logs exported to: {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Signal Stability Gateway - Neural Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m ssg.main run --duration 60
    python -m ssg.main run --headless --duration 30
    python -m ssg.main test --validate-performance
    python -m ssg.main test --duration 10
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run SSG pipeline')
    run_parser.add_argument(
        '--duration', '-d',
        type=float,
        default=60.0,
        help='Duration in seconds (0 for indefinite)',
    )
    run_parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without dashboard',
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
        help='Export logs to JSON file on exit',
    )

    # Test command
    test_parser = subparsers.add_parser('test', help='Run test harness')
    test_parser.add_argument(
        '--duration', '-d',
        type=float,
        default=10.0,
        help='Test duration in seconds',
    )
    test_parser.add_argument(
        '--validate-performance',
        action='store_true',
        help='Run performance validation',
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

    args = parser.parse_args()

    if args.command == 'run':
        ssg = SignalStabilityGateway(
            n_channels=args.channels,
            seed=args.seed,
        )

        ssg.run(
            duration_sec=args.duration,
            headless=args.headless,
        )

        if args.export_logs:
            ssg.export_logs(args.export_logs)

    elif args.command == 'test':
        harness = TestHarness(seed=args.seed)

        if args.validate_performance:
            print("Running performance validation...")
            results = harness.validate_performance(
                target_latency_ms=5.0,
                duration_sec=args.duration,
            )

            print(f"\nPerformance Validation: {'PASSED' if results['passed'] else 'FAILED'}")
            print(f"  Average Latency: {results['avg_latency_ms']:.2f}ms")
            print(f"  Max Latency: {results['max_latency_ms']:.2f}ms")
            print(f"  Target: {results['target_latency_ms']:.2f}ms")
            print(f"  Batches/sec: {results['batches_per_second']:.1f}")

            sys.exit(0 if results['passed'] else 1)

        else:
            print(f"Running test harness for {args.duration}s...")
            results = harness.run(
                duration_sec=args.duration,
                inject_artifacts=not args.no_artifacts,
            )

            print(f"\nTest Results:")
            print(f"  Total Batches: {results.total_batches}")
            print(f"  Duration: {results.total_duration_sec:.2f}s")
            print(f"  Avg Latency: {results.avg_latency_ms:.2f}ms")
            print(f"  Max Latency: {results.max_latency_ms:.2f}ms")
            print(f"  Viable Channels: {results.min_viable_channels}-{results.max_viable_channels} (avg: {results.avg_viable_channels:.0f})")
            print(f"  Artifacts Injected: {results.total_artifacts_injected}")
            print(f"  SNR: min={results.snr_distribution['min']:.2f}, max={results.snr_distribution['max']:.2f}, mean={results.snr_distribution['mean']:.2f}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
