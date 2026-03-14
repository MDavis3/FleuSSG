"""Shared dtypes and dataclasses used across the SSG pipeline."""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .array_types import BoolVector, FloatMatrix, FloatVector
from .constants import N_CHANNELS


# =============================================================================
# NUMPY DTYPES
# =============================================================================

def make_channel_frame_dtype(n_channels: int = N_CHANNELS) -> np.dtype:
    """Create a structured dtype for one timestamped multi-channel sample."""

    return np.dtype(
        [
            ("timestamp_us", np.uint64),
            ("samples", np.float32, (n_channels,)),
        ]
    )


ChannelFrameDtype = make_channel_frame_dtype()

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


def _expect_matrix(name: str, value: np.ndarray) -> None:
    """Require a matrix input for batch-oriented arrays."""
    if value.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {value.ndim}D")


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
        _expect_matrix("raw_unfiltered", self.raw_unfiltered)
        _expect_matrix("lfp", self.lfp)
        _expect_matrix("spikes", self.spikes)
        expected_shape = self.raw_unfiltered.shape
        if self.lfp.shape != expected_shape:
            raise ValueError(
                f"lfp must match raw_unfiltered shape {expected_shape}, got {self.lfp.shape}"
            )
        if self.spikes.shape != expected_shape:
            raise ValueError(
                f"spikes must match raw_unfiltered shape {expected_shape}, got {self.spikes.shape}"
            )
        _expect_shape("artifact_flags", self.artifact_flags, (expected_shape[1],))


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
        n_channels = len(self.viability_mask)
        _expect_shape("snr", self.snr, (n_channels,))
        _expect_shape("firing_rate_hz", self.firing_rate_hz, (n_channels,))
        _expect_shape("isi_violation_rate", self.isi_violation_rate, (n_channels,))
        _expect_shape("impedance_kohm", self.impedance_kohm, (n_channels,))
        _expect_shape("viability_mask", self.viability_mask, (n_channels,))
        if not 0 <= self.viable_channel_count <= n_channels:
            raise ValueError(
                "viable_channel_count must be within the channel count contract"
            )

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
