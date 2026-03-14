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

    notch_sos_list: list[np.ndarray]
    lfp_sos: np.ndarray
    spike_sos: np.ndarray


@dataclass
class SOSFilterState:
    """Streaming SOS state for the notch, LFP, and spike filters."""

    notch_zi: list[np.ndarray]
    lfp_zi: FloatMatrix
    spike_zi: FloatMatrix


def design_filter_bank(sample_rate_hz: int) -> FilterBank:
    """Precompute the filter coefficients for the sanitization pipeline."""

    nyquist = sample_rate_hz / 2.0
    notch_sos_list: list[np.ndarray] = []
    for freq in NOTCH_FREQUENCIES_HZ:
        if freq >= nyquist:
            continue
        b, a = signal.iirnotch(freq, NOTCH_QUALITY_FACTOR, sample_rate_hz)
        notch_sos_list.append(
            np.array(
                [[b[0], b[1], b[2], a[0], a[1], a[2]]],
                dtype=np.float32,
            )
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
        notch_sos_list=notch_sos_list,
        lfp_sos=lfp_sos,
        spike_sos=spike_sos,
    )


def init_filter_state(filter_bank: FilterBank, n_channels: int) -> SOSFilterState:
    """Initialize SOS streaming state for each filter stage."""

    notch_zi = []
    for sos in filter_bank.notch_sos_list:
        zi = np.asarray(signal.sosfilt_zi(sos), dtype=np.float32)
        notch_zi.append(np.tile(zi[:, :, np.newaxis], (1, 1, n_channels)).copy())

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
        notch_zi=notch_zi,
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
    for index, sos in enumerate(filter_bank.notch_sos_list):
        notched, filter_state.notch_zi[index] = signal.sosfilt(
            sos,
            notched,
            axis=0,
            zi=filter_state.notch_zi[index],
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
