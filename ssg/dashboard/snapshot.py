"""Capture a textual dashboard snapshot with numeric specs."""

import time
from dataclasses import dataclass

from rich.console import Console

from ..core.constants import (
    ISI_VIOLATION_LIMIT,
    LATENCY_WARNING_THRESHOLD_MS,
    MILLISECONDS_PER_SECOND,
    REGION_DEFINITIONS,
    SNR_THRESHOLD,
)
from ..core.pipeline_runtime import PipelineRuntime, PipelineRuntimeConfig
from ..ingestion.mock_telemetry import MockTelemetryConfig
from .cli import Dashboard
from .messages import (
    format_artifact_event_message,
    format_latency_warning_message,
)


@dataclass(frozen=True)
class RegionSnapshot:
    """Per-region viability details from a dashboard snapshot."""

    name: str
    viable: int
    total: int
    viability_pct: float


@dataclass(frozen=True)
class DashboardSnapshot:
    """Rendered dashboard text plus the numeric specs behind it."""

    rendered_text: str
    seed: int | None
    batches_rendered: int
    channels: int
    sample_rate_hz: int
    batch_size: int
    batch_duration_ms: float
    snr_threshold: float
    isi_violation_limit_pct: float
    viable_channels: int
    total_channels: int
    mean_snr: float
    min_snr: float
    max_snr: float
    mean_isi_violation_pct: float
    max_isi_violation_pct: float
    mean_firing_rate_hz: float
    min_firing_rate_hz: float
    max_firing_rate_hz: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    latest_timestamp_us: int
    regions: tuple[RegionSnapshot, ...]
    event_log: tuple[str, ...]

    def to_text(self) -> str:
        """Format the snapshot for CLI output."""

        lines = [
            "=== DASHBOARD SNAPSHOT ===",
            self.rendered_text.rstrip(),
            "",
            "=== NUMERIC SPECS ===",
            f"seed: {self.seed}",
            f"batches_rendered: {self.batches_rendered}",
            f"channels: {self.channels}",
            f"sample_rate_hz: {self.sample_rate_hz}",
            f"batch_size: {self.batch_size}",
            f"batch_duration_ms: {self.batch_duration_ms:.1f}",
            f"snr_threshold: {self.snr_threshold}",
            f"isi_violation_limit_pct: {self.isi_violation_limit_pct:.2f}",
            f"viable_channels: {self.viable_channels}/{self.total_channels}",
            f"mean_snr: {self.mean_snr:.4f}",
            f"min_snr: {self.min_snr:.4f}",
            f"max_snr: {self.max_snr:.4f}",
            f"mean_isi_violation_pct: {self.mean_isi_violation_pct:.4f}",
            f"max_isi_violation_pct: {self.max_isi_violation_pct:.4f}",
            f"mean_firing_rate_hz: {self.mean_firing_rate_hz:.4f}",
            f"min_firing_rate_hz: {self.min_firing_rate_hz:.4f}",
            f"max_firing_rate_hz: {self.max_firing_rate_hz:.4f}",
            f"avg_latency_ms: {self.avg_latency_ms:.4f}",
            f"min_latency_ms: {self.min_latency_ms:.4f}",
            f"max_latency_ms: {self.max_latency_ms:.4f}",
            f"latest_timestamp_us: {self.latest_timestamp_us}",
            "region_viability:",
        ]
        lines.extend(
            f"  {region.name}: viable={region.viable}/{region.total} pct={region.viability_pct:.2f}"
            for region in self.regions
        )
        lines.append("event_log:")
        lines.extend(f"  {event}" for event in self.event_log)
        return "\n".join(lines)


def capture_dashboard_snapshot(
    seed: int | None = 42,
    batches: int = 10,
    width: int = 160,
) -> DashboardSnapshot:
    """Render a real dashboard frame and collect the numeric specs behind it."""

    if batches <= 0:
        raise ValueError("batches must be positive")

    runtime = PipelineRuntime(
        config=PipelineRuntimeConfig(
            telemetry_config=MockTelemetryConfig(seed=seed),
        )
    )
    dashboard = Dashboard()
    dashboard.add_event(
        "SYSTEM",
        f"Started: {runtime.config.n_channels} channels @ {runtime.config.sample_rate_hz}Hz",
        "INFO",
    )

    latencies: list[float] = []
    latest_metrics = None
    latest_batch = None
    for batch_index in range(batches):
        batch_start = time.perf_counter()
        latest_batch = runtime.process_next_batch()
        latency_ms = (time.perf_counter() - batch_start) * MILLISECONDS_PER_SECOND
        latest_metrics = latest_batch.metrics
        dashboard.update(latest_metrics, latency_ms)
        latencies.append(latency_ms)

        if latest_batch.sanitized_frame.artifact_flags.any():
            affected = int(latest_batch.sanitized_frame.artifact_flags.sum())
            dashboard.add_event(
                "ARTIFACT",
                format_artifact_event_message(affected),
                "WARNING",
            )
        if latency_ms > LATENCY_WARNING_THRESHOLD_MS:
            dashboard.add_event(
                "LATENCY",
                format_latency_warning_message(batch_index, latency_ms),
                "WARNING",
            )

    assert latest_metrics is not None
    assert latest_batch is not None
    console = Console(record=True, width=width)
    console.print(dashboard._create_layout())

    regions = tuple(
        RegionSnapshot(
            name=region_name,
            viable=viable,
            total=total,
            viability_pct=pct * 100,
        )
        for region_name, (start, end) in REGION_DEFINITIONS.items()
        for viable, total, pct in [latest_metrics.get_region_viability(start, end)]
    )
    event_log = tuple(
        f"[{event.timestamp}] {event.severity} {event.event_type}: {event.message}"
        for event in dashboard._event_log
    )
    batch_duration_ms = (
        runtime.config.batch_size
        / runtime.config.sample_rate_hz
        * MILLISECONDS_PER_SECOND
    )
    return DashboardSnapshot(
        rendered_text=console.export_text(),
        seed=seed,
        batches_rendered=batches,
        channels=runtime.config.n_channels,
        sample_rate_hz=runtime.config.sample_rate_hz,
        batch_size=runtime.config.batch_size,
        batch_duration_ms=batch_duration_ms,
        snr_threshold=SNR_THRESHOLD,
        isi_violation_limit_pct=ISI_VIOLATION_LIMIT * 100,
        viable_channels=latest_metrics.viable_channel_count,
        total_channels=runtime.config.n_channels,
        mean_snr=float(latest_metrics.snr.mean()),
        min_snr=float(latest_metrics.snr.min()),
        max_snr=float(latest_metrics.snr.max()),
        mean_isi_violation_pct=float(latest_metrics.isi_violation_rate.mean()) * 100,
        max_isi_violation_pct=float(latest_metrics.isi_violation_rate.max()) * 100,
        mean_firing_rate_hz=float(latest_metrics.firing_rate_hz.mean()),
        min_firing_rate_hz=float(latest_metrics.firing_rate_hz.min()),
        max_firing_rate_hz=float(latest_metrics.firing_rate_hz.max()),
        avg_latency_ms=sum(latencies) / len(latencies),
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        latest_timestamp_us=latest_batch.metrics.timestamp_us,
        regions=regions,
        event_log=event_log,
    )
