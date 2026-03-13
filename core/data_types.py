"""
Data Types for Signal Stability Gateway

Defines NumPy dtypes and dataclasses for the SSG pipeline.
All structures are optimized for vectorized processing.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
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

@dataclass
class SanitizedFrame:
    """
    Output from SanitizationLayer.

    Contains separated LFP and spike bands with artifact detection flags.
    All arrays are shape (N, N_CHANNELS) where N is batch size.
    """
    timestamp_us: int
    raw_unfiltered: np.ndarray    # shape (N, 1024) - Original signal preserved
    lfp: np.ndarray               # shape (N, 1024) - Low-frequency (<300Hz)
    spikes: np.ndarray            # shape (N, 1024) - High-frequency (300-3000Hz)
    artifact_flags: np.ndarray    # shape (1024,) - Per-channel artifact detection

    def __post_init__(self):
        """Validate array shapes."""
        assert self.lfp.shape[1] == N_CHANNELS, f"LFP must have {N_CHANNELS} channels"
        assert self.spikes.shape[1] == N_CHANNELS, f"Spikes must have {N_CHANNELS} channels"
        assert self.artifact_flags.shape == (N_CHANNELS,), "Artifact flags must be (1024,)"


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
    snr: np.ndarray                    # shape (1024,) - Signal-to-Noise Ratio
    firing_rate_hz: np.ndarray         # shape (1024,) - Spikes/sec (for TN-VAE)
    isi_violation_rate: np.ndarray     # shape (1024,) - ISI violation percentage
    impedance_kohm: np.ndarray         # shape (1024,) - Electrode impedance
    viability_mask: np.ndarray         # shape (1024,), dtype=bool
    viable_channel_count: int          # Count of viable channels

    def __post_init__(self):
        """Validate array shapes and compute derived fields."""
        assert self.snr.shape == (N_CHANNELS,)
        assert self.firing_rate_hz.shape == (N_CHANNELS,)
        assert self.isi_violation_rate.shape == (N_CHANNELS,)
        assert self.impedance_kohm.shape == (N_CHANNELS,)
        assert self.viability_mask.shape == (N_CHANNELS,)

    def get_region_viability(self, start: int, end: int) -> tuple:
        """Get viability stats for a region slice."""
        region_mask = self.viability_mask[start:end]
        viable = int(np.sum(region_mask))
        total = end - start
        return viable, total, viable / total if total > 0 else 0.0


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
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "timestamp_us": self.timestamp_us,
            "event_type": self.event_type,
            "channel_id": self.channel_id,
            "severity": self.severity,
            "message": self.message,
            "metadata": self.metadata,
        }
