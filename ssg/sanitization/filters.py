"""Filter design and streaming helpers for sanitization."""

from dataclasses import dataclass

import numpy as np
from scipy import signal

from ..core.array_types import FloatMatrix
from ..core.constants import (
    BUTTERWORTH_ORDER,
    LFP_CUTOFF_HZ,
    NOTCH_FREQUENCIES_HZ,
    NOTCH_QUALITY_FACTOR,
    SPIKE_HIGHCUT_HZ,
    SPIKE_LOWCUT_HZ,
)


@dataclass
class FilterBank:
    """Precomputed filter coefficients used by the streaming layer."""

    notch_sos: np.ndarray
    lfp_sos: np.ndarray
    spike_sos: np.ndarray


@dataclass
class SOSFilterState:
    """Streaming SOS state for the notch, LFP, and spike filters."""

    notch_zi: np.ndarray
    lfp_zi: np.ndarray
    spike_zi: np.ndarray


def design_filter_bank(sample_rate_hz: int) -> FilterBank:
    """Precompute the filter coefficients for the sanitization pipeline."""

    nyquist = sample_rate_hz / 2.0
    notch_sections: list[np.ndarray] = []
    for freq in NOTCH_FREQUENCIES_HZ:
        if freq >= nyquist:
            continue
        b, a = signal.iirnotch(freq, NOTCH_QUALITY_FACTOR, sample_rate_hz)
        notch_sections.append(
            np.array(
                [[b[0], b[1], b[2], a[0], a[1], a[2]]],
                dtype=np.float32,
            )
        )
    notch_sos = (
        np.vstack(notch_sections).astype(np.float32)
        if notch_sections
        else np.empty((0, 6), dtype=np.float32)
    )

    lfp_sos = signal.butter(
        BUTTERWORTH_ORDER,
        LFP_CUTOFF_HZ / nyquist,
        btype="low",
        output="sos",
    ).astype(np.float32)
    spike_sos = signal.butter(
        BUTTERWORTH_ORDER,
        [SPIKE_LOWCUT_HZ / nyquist, SPIKE_HIGHCUT_HZ / nyquist],
        btype="band",
        output="sos",
    ).astype(np.float32)
    return FilterBank(
        notch_sos=notch_sos,
        lfp_sos=lfp_sos,
        spike_sos=spike_sos,
    )


def init_filter_state(filter_bank: FilterBank, n_channels: int) -> SOSFilterState:
    """Initialize SOS streaming state for each filter stage."""

    notch_zi = np.tile(
        np.asarray(signal.sosfilt_zi(filter_bank.notch_sos), dtype=np.float32)[
            :,
            :,
            np.newaxis,
        ],
        (1, 1, n_channels),
    )

    lfp_zi = np.tile(
        np.asarray(signal.sosfilt_zi(filter_bank.lfp_sos), dtype=np.float32)[
            :,
            :,
            np.newaxis,
        ],
        (1, 1, n_channels),
    )
    spike_zi = np.tile(
        np.asarray(signal.sosfilt_zi(filter_bank.spike_sos), dtype=np.float32)[
            :,
            :,
            np.newaxis,
        ],
        (1, 1, n_channels),
    )
    return SOSFilterState(
        notch_zi=notch_zi.copy(),
        lfp_zi=lfp_zi.copy(),
        spike_zi=spike_zi.copy(),
    )


def apply_filter_bank(
    samples: FloatMatrix,
    filter_bank: FilterBank,
    filter_state: SOSFilterState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a batch through the notch, LFP, and spike filters."""

    notched = np.asarray(samples, dtype=np.float32)
    if filter_bank.notch_sos.size > 0:
        notched, filter_state.notch_zi = signal.sosfilt(
            filter_bank.notch_sos,
            notched,
            axis=0,
            zi=filter_state.notch_zi,
        )

    lfp, filter_state.lfp_zi = signal.sosfilt(
        filter_bank.lfp_sos,
        notched,
        axis=0,
        zi=filter_state.lfp_zi,
    )
    spikes, filter_state.spike_zi = signal.sosfilt(
        filter_bank.spike_sos,
        notched,
        axis=0,
        zi=filter_state.spike_zi,
    )
    return (
        np.asarray(notched, dtype=np.float32),
        np.asarray(lfp, dtype=np.float32),
        np.asarray(spikes, dtype=np.float32),
    )
