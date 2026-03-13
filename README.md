# Signal Stability Gateway (SSG)

Mission-critical middleware for bridging high-density neural electrode arrays with AI orchestration and foundation models.

## Overview

SSG is a real-time signal processing pipeline designed for 1,024-channel neural recordings. It provides:

- **Signal Sanitization**: 4th-order Butterworth filters for LFP/spike band separation, 60Hz notch filtering
- **Artifact Rejection**: Tanh-scaling with rolling baseline sigma
- **Channel Validation**: Real-time SNR, ISI violations, and viability scoring
- **FDA Traceability**: All thresholds documented for regulatory compliance

## Architecture

```
IngestionEngine (Ring Buffer)
       │
       ▼
SanitizationLayer (DSP Filters + Artifact Rejection)
       │
       ▼
ValidationEngine (SNR, ISI, Viability Mask)
       │
       ▼
Dashboard (Rich CLI) / Export (JSON/CSV)
```

## Installation

```bash
pip install numpy scipy rich
```

## Usage

### Run with Dashboard
```bash
python -m ssg.main run --duration 60
```

### Run Headless
```bash
python -m ssg.main run --headless --duration 30
```

### Run Tests
```bash
python -m ssg.main test --duration 10
```

### Validate Performance
```bash
python -m ssg.main test --validate-performance
```

## Module Structure

```
ssg/
├── core/
│   ├── constants.py      # FDA-traceable thresholds
│   ├── data_types.py     # NumPy dtypes and dataclasses
│   └── ring_buffer.py    # Zero-copy ring buffer
├── ingestion/
│   ├── engine.py         # IngestionEngine
│   └── mock_telemetry.py # Synthetic signal generator
├── sanitization/
│   └── layer.py          # DSP filters + artifact rejection
├── validation/
│   └── engine.py         # SNR, ISI, viability scoring
├── dashboard/
│   └── cli.py            # Rich CLI visualization
├── simulation/
│   ├── noise_models.py   # Biological artifact injection
│   └── test_harness.py   # End-to-end testing
├── logging/
│   ├── audit_logger.py   # FDA audit trail
│   ├── event_types.py    # Enumerated events
│   └── exporters.py      # JSON/CSV export
└── main.py               # CLI entry point
```

## Key Specifications

| Parameter | Value | Reference |
|-----------|-------|-----------|
| Channels | 1,024 | High-density array spec |
| Sample Rate | 20 kHz | Clinical standard |
| SNR Threshold | >= 4.0 | SpikeAgent benchmark |
| ISI Violation Limit | < 1.5% | Biophysical refractory period |
| Impedance Range | 50-3000 kOhm | Clinical electrode spec |

## Signal Processing Pipeline

### Sanitization Layer
1. **60Hz Notch Filter**: IIR notch at 60, 120, 180 Hz (Q=30)
2. **LFP Extraction**: 4th-order Butterworth lowpass, fc=300Hz
3. **Spike Extraction**: 4th-order Butterworth bandpass, 300-3000Hz
4. **Artifact Detection**: Tanh-scaling at 4σ with rolling EMA baseline

### Validation Engine
- **SNR**: Rolling EMA (α=0.01), MAD-based noise estimation
- **ISI Violations**: Vectorized using `np.lexsort` + `np.diff` + `np.bincount`
- **Firing Rate**: O(n_spikes) computation for downstream models
- **Viability Mask**: Boolean mask combining all criteria

## Viability Criteria

A channel is marked as **viable** if:
```python
viability_mask = (
    (snr >= 4.0) &
    (isi_violation_rate < 0.015) &
    (impedance >= 50) &
    (impedance <= 3000) &
    (~artifact_flags)
)
```

## Simulation & Testing

The simulation module provides biological noise injection:

- **Jaw Clench**: EMG burst, 50-500ms, ~30% channels
- **Electrode Drift**: DC shift, ~2s, ~5% channels
- **Motion Spike**: Sharp transient, ~10ms, ~80% channels

```python
from ssg.simulation.test_harness import TestHarness

harness = TestHarness(seed=42)
results = harness.run(duration_sec=10.0, inject_artifacts=True)
print(f"Viable channels: {results.avg_viable_channels}")
```

## FDA Audit Logging

All events are logged with timestamps for regulatory compliance:

```python
from ssg.logging import AuditLogger, JSONExporter

logger = AuditLogger()
# ... processing ...
JSONExporter.export_logger(logger, "audit_log.json")
```

## Dependencies

- `numpy >= 1.24.0`
- `scipy >= 1.10.0`
- `rich >= 13.0.0`

## License

MIT License
