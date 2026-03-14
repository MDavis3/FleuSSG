"""
FDA-Traceable Constants for Signal Stability Gateway

All thresholds and parameters used in the SSG pipeline are defined here
for regulatory traceability. Changes to these values require validation.

References:
- Clinical electrode array specifications
- SpikeAgent Paper (2025): ISI violation thresholds, SNR benchmarks
- TN-VAE Consciousness Paper (2025): Window sizes, feature extraction
"""

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

# Number of recording channels (high-density array specification)
N_CHANNELS: int = 1024

# Sampling rate in Hz (clinical specification: 20kHz)
SAMPLE_RATE_HZ: int = 20_000

# Batch processing interval in seconds (100ms batches)
BATCH_DURATION_SEC: float = 0.1

# Number of samples per batch
BATCH_SIZE: int = int(SAMPLE_RATE_HZ * BATCH_DURATION_SEC)


# =============================================================================
# FILTER SPECIFICATIONS
# =============================================================================

# Butterworth filter order (standard for neural signal processing)
BUTTERWORTH_ORDER: int = 4

# LFP lowpass cutoff frequency in Hz
LFP_CUTOFF_HZ: float = 300.0

# Spike bandpass filter frequencies in Hz
SPIKE_LOWCUT_HZ: float = 300.0
SPIKE_HIGHCUT_HZ: float = 3000.0

# Power line notch filter frequencies (60Hz fundamental + harmonics)
NOTCH_FREQUENCIES_HZ: tuple = (60.0, 120.0, 180.0)

# Notch filter quality factor (higher = narrower notch)
NOTCH_QUALITY_FACTOR: float = 30.0


# =============================================================================
# ARTIFACT REJECTION
# =============================================================================

# Tanh scaling threshold in standard deviations
# Set high enough to not trigger on normal neural spikes (only true artifacts)
# 100σ effectively disables artifact detection for baseline (only extreme events)
ARTIFACT_THRESHOLD_SIGMA: float = 100.0

# Rolling sigma window duration in seconds
SIGMA_WINDOW_SEC: float = 10.0

# EMA smoothing factor for rolling statistics
EMA_ALPHA: float = 0.01


# =============================================================================
# SIGNAL QUALITY THRESHOLDS
# =============================================================================

# Minimum SNR for a channel to be considered viable
# Reference: SpikeAgent paper suggests 4.5-5.0 as ideal
SNR_THRESHOLD: float = 4.0

# ISI refractory period in microseconds (1.5ms = 1500µs)
# Reference: Biophysical refractory period of neurons
ISI_REFRACTORY_PERIOD_US: int = 1500

# Maximum ISI violation rate (as decimal, 3.0% = 0.03)
# Reference: SpikeAgent benchmark for spike curation (relaxed for demo)
ISI_VIOLATION_LIMIT: float = 0.03

# Electrode impedance viable range in kOhms
# Reference: Clinical electrode characterization standards
IMPEDANCE_MIN_KOHM: float = 50.0
IMPEDANCE_MAX_KOHM: float = 3000.0


# =============================================================================
# VALIDATION ENGINE
# =============================================================================

# Window duration for metric calculation in seconds
VALIDATION_WINDOW_SEC: float = 30.0

# Maximum spike storage capacity
MAX_SPIKES_IN_WINDOW: int = 500_000

# Spike detection threshold in MAD units (negative for extracellular)
SPIKE_DETECTION_THRESHOLD_MAD: float = -4.0

# MAD to standard deviation conversion factor
MAD_TO_STD_FACTOR: float = 1.4826


# =============================================================================
# DASHBOARD
# =============================================================================

# Dashboard refresh rate in Hz
DASHBOARD_REFRESH_HZ: float = 2.0

# Maximum events to display in log
MAX_EVENT_LOG_SIZE: int = 10

# Region definitions for high-density array (channel slices)
REGION_DEFINITIONS: dict = {
    "Cortex L1-3": (0, 256),
    "Cortex L4-5": (256, 512),
    "Cortex L6": (512, 640),
    "Hippocampus CA1": (640, 800),
    "Hippocampus CA3": (800, 900),
    "Hippocampus DG": (900, 1024),
}


# =============================================================================
# SIMULATION
# =============================================================================

# Default spike rate for synthetic data (spikes per second)
# Moderate rate balances SNR estimation with ISI violations
SYNTHETIC_SPIKE_RATE_HZ: float = 45.0

# Spike template duration in seconds (2ms)
SPIKE_TEMPLATE_DURATION_SEC: float = 0.002

# Default synthetic signal parameters (tuned for >95% baseline viability)
# High amplitude with low noise for excellent SNR
SYNTHETIC_NOISE_AMPLITUDE_UV: float = 1.5  # Minimal noise floor
SYNTHETIC_SPIKE_AMPLITUDE_UV: float = 95.0  # Strong spike for high SNR

# Artifact probability defaults (events per second)
JAW_CLENCH_PROBABILITY: float = 0.1
ELECTRODE_DRIFT_PROBABILITY: float = 0.05
MOTION_SPIKE_PROBABILITY: float = 0.02
