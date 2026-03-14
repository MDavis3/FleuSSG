"""
Data Types for Signal Stability Gateway

Defines NumPy dtypes and dataclasses for the SSG pipeline.
All structures are optimized for vectorized processing.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional
from .array_types import BoolVector, FloatMatrix, FloatVector
from .constants import N_CHANNELS


# =============================================================================
# NUMPY DTYPES
# =============================================================================

# Primary data frame dtype - one sample across all channels
ChannelFrameDtype = np.dtype([
    ('timestamp_us', np.uint64),              # Microsecond timestamp
    ('samples', np.float32, (N_CHANNELS,)),   # 1024 channels of raw voltage
])

# Viability mask - boolean array for downstream AI
ViabilityMask = np.ndarray  # shape (N_CHANNELS,), dtype=bool


# =============================================================================
# DATACLASSES
# =============================================================================


def _expect_shape(name: str, value: np.ndarray, expected_shape: tuple[int, ...]) -> None:
    """Raise a stable error when an array does not match its contract."""
    if value.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {value.shape}"
        )


@dataclass
class SanitizedFrame:
    """
    Output from SanitizationLayer.

    Contains separated LFP and spike bands with artifact detection flags.
    All arrays are shape (N, N_CHANNELS) where N is batch size.
    """
    timestamp_us: int
    raw_unfiltered: FloatMatrix   # shape (N, 1024) - Original signal preserved
    lfp: FloatMatrix              # shape (N, 1024) - Low-frequency (<300Hz)
    spikes: FloatMatrix           # shape (N, 1024) - High-frequency (300-3000Hz)
    artifact_flags: BoolVector    # shape (1024,) - Per-channel artifact detection

    def __post_init__(self):
        """Validate array shapes."""
        if self.lfp.ndim != 2 or self.lfp.shape[1] != N_CHANNELS:
            raise ValueError(
                f"lfp must have shape (batch_size, {N_CHANNELS}), got {self.lfp.shape}"
            )
        if self.spikes.ndim != 2 or self.spikes.shape[1] != N_CHANNELS:
            raise ValueError(
                f"spikes must have shape (batch_size, {N_CHANNELS}), got {self.spikes.shape}"
            )
        if self.raw_unfiltered.shape != self.lfp.shape:
            raise ValueError(
                "raw_unfiltered must match the lfp batch shape"
            )
        _expect_shape("artifact_flags", self.artifact_flags, (N_CHANNELS,))


@dataclass
class ChannelMetrics:
    """
    Per-batch metrics output from ValidationEngine.

    Used by:
    - MockDashboard for visualization
    - TN-VAE for firing_rate_hz input
    - SpikeAgent for viability_mask
    """
    timestamp_us: int
    snr: FloatVector                   # shape (1024,) - Signal-to-Noise Ratio
    firing_rate_hz: FloatVector        # shape (1024,) - Spikes/sec (for TN-VAE)
    isi_violation_rate: FloatVector    # shape (1024,) - ISI violation percentage
    impedance_kohm: FloatVector        # shape (1024,) - Electrode impedance
    viability_mask: BoolVector         # shape (1024,), dtype=bool
    viable_channel_count: int          # Count of viable channels

    def __post_init__(self):
        """Validate array shapes and compute derived fields."""
        _expect_shape("snr", self.snr, (N_CHANNELS,))
        _expect_shape("firing_rate_hz", self.firing_rate_hz, (N_CHANNELS,))
        _expect_shape("isi_violation_rate", self.isi_violation_rate, (N_CHANNELS,))
        _expect_shape("impedance_kohm", self.impedance_kohm, (N_CHANNELS,))
        _expect_shape("viability_mask", self.viability_mask, (N_CHANNELS,))

    def get_region_viability(self, start: int, end: int) -> tuple[int, int, float]:
        """Get viability stats for a region slice."""
        region_mask = self.viability_mask[start:end]
        viable = int(np.sum(region_mask))
        total = end - start
        return viable, total, viable / total if total > 0 else 0.0


@dataclass(frozen=True)
class RegionMetrics:
    """Aggregated channel-quality metrics for a channel slice."""
    viable_count: int
    total_count: int
    viability_pct: float
    mean_snr: float
    mean_firing_rate_hz: float


@dataclass
class AuditEvent:
    """
    Event structure for FDA audit logging.
    """
    timestamp_us: int
    event_type: str
    channel_id: Optional[int] = None
    severity: str = "INFO"
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON export."""
        return {
            "timestamp_us": self.timestamp_us,
            "event_type": self.event_type,
            "channel_id": self.channel_id,
            "severity": self.severity,
            "message": self.message,
            "metadata": self.metadata,
        }
