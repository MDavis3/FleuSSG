import numpy as np
import pytest

from ssg.core.constants import N_CHANNELS
from ssg.core.data_types import ChannelMetrics, SanitizedFrame
from ssg.validation.engine import ValidationEngine


def _make_frame(start_timestamp_us: int, batch_size: int, spike_positions: list[int]):
    spikes = np.zeros((batch_size, N_CHANNELS), dtype=np.float32)
    for position in spike_positions:
        spikes[position - 1:position + 2, 0] = np.array([0.0, -10.0, 0.0], dtype=np.float32)

    timestamps = start_timestamp_us + (
        np.arange(batch_size, dtype=np.uint64) * 1000
    )
    frame = SanitizedFrame(
        timestamp_us=int(timestamps[0]),
        raw_unfiltered=spikes.copy(),
        lfp=spikes.copy(),
        spikes=spikes.copy(),
        artifact_flags=np.zeros(N_CHANNELS, dtype=bool),
    )
    return frame, timestamps


def test_validation_engine_trims_spikes_to_validation_window():
    engine = ValidationEngine(sample_rate_hz=1000)

    first_frame, first_timestamps = _make_frame(0, 100, [10, 20, 30])
    first_metrics = engine.process(first_frame, first_timestamps)

    second_frame, second_timestamps = _make_frame(31_000_000, 100, [10])
    second_metrics = engine.process(second_frame, second_timestamps)

    assert first_metrics.firing_rate_hz[0] == pytest.approx(30.0)
    assert second_metrics.firing_rate_hz[0] == pytest.approx(5.0)


def test_sanitized_frame_rejects_mismatched_batch_shapes():
    with pytest.raises(ValueError):
        SanitizedFrame(
            timestamp_us=0,
            raw_unfiltered=np.zeros((4, N_CHANNELS), dtype=np.float32),
            lfp=np.zeros((3, N_CHANNELS), dtype=np.float32),
            spikes=np.zeros((3, N_CHANNELS), dtype=np.float32),
            artifact_flags=np.zeros(N_CHANNELS, dtype=bool),
        )


def test_channel_metrics_accept_custom_channel_width():
    metrics = ChannelMetrics(
        timestamp_us=0,
        snr=np.ones(8, dtype=np.float32),
        firing_rate_hz=np.zeros(8, dtype=np.float32),
        isi_violation_rate=np.zeros(8, dtype=np.float32),
        impedance_kohm=np.full(8, 1000.0, dtype=np.float32),
        viability_mask=np.array([True, False, True, True, False, True, False, True]),
        viable_channel_count=5,
    )

    assert metrics.get_region_viability(0, 4) == (3, 4, 0.75)
