# Signal Stability Gateway (SSG)

Signal Stability Gateway is a Python package for simulating, sanitizing, validating, and inspecting high-density neural recordings. The current repository is a research-oriented prototype: it models a streaming pipeline, ships a Rich dashboard, and includes synthetic telemetry plus artifact injection for local experimentation.

## What It Includes

- `ingestion/`: batch ingestion and mock telemetry for synthetic recordings
- `sanitization/`: notch, low-pass, and spike-band filtering plus artifact flags
- `validation/`: rolling signal-quality metrics and channel viability scoring
- `dashboard/`: a CLI monitor for live runs
- `simulation/`: artifact models and an end-to-end harness for local validation
- `logging/`: timestamped audit events plus JSON and CSV exporters

## Installation

```bash
pip install -e .
```

For development extras:

```bash
pip install -e .[dev]
```

## Usage

Run the demo pipeline with the CLI entry point:

```bash
ssg run --duration 60
```

Run the same flow headlessly:

```bash
ssg run --headless --duration 30
```

Exercise the simulation harness:

```bash
ssg test --duration 10
```

Validate average batch latency against the configured target:

```bash
ssg test --validate-performance
```

## Architecture

```text
MockTelemetry or external source
  -> IngestionEngine
  -> SanitizationLayer
  -> ValidationEngine
  -> Dashboard / JSON export / CSV export
```

The default configuration models 1,024 channels at 20 kHz with rolling SNR, ISI, impedance, and artifact-based viability checks. Thresholds live in `core/constants.py` and should be treated as prototype defaults unless separately validated for a deployment context.

## Example

```python
from ssg.audit.audit_logger import AuditLogger
from ssg.audit.exporters import JSONExporter
from ssg.bench.test_harness import TestHarness

harness = TestHarness(seed=42)
results = harness.run(duration_sec=10.0, inject_artifacts=True)

logger = AuditLogger()
logger.log_batch(
    batch_id=0,
    viable_count=int(results.avg_viable_channels),
    latency_ms=results.avg_latency_ms,
)
JSONExporter.export_logger(logger, "audit-log.json")
```

## Development

The repository includes targeted pytest coverage for core buffers, dashboard state, audit logging, exporters, validation, and CLI orchestration:

```bash
pytest -q
```

## License

MIT License
