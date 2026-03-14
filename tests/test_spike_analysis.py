import numpy as np
import pytest

from ssg.validation.metrics import summarize_spike_band
from ssg.validation.spike_analysis import (
    SpikeBuffer,
    compute_firing_rate,
    compute_isi_violations,
    detect_spikes,
)


def test_spike_buffer_trim_before_discards_old_events():
    buffer = SpikeBuffer.create(capacity=8)
    buffer.add_spikes(
        np.array([0, 1, 0], dtype=np.uint16),
        np.array([100, 200, 300], dtype=np.uint64),
        np.array([-4.0, -5.0, -6.0], dtype=np.float32),
    )

    buffer.trim_before(250)
    channels, timestamps, amplitudes = buffer.get_valid()

    assert channels.tolist() == [0]
    assert timestamps.tolist() == [300]
    assert amplitudes.tolist() == pytest.approx([-6.0])


def test_detect_spikes_filters_same_channel_hits_inside_refractory_window():
    spike_band = np.zeros((5, 1), dtype=np.float32)
    spike_band[:, 0] = np.array([0.0, -10.0, 0.0, -9.0, 0.0], dtype=np.float32)
    timestamps = np.arange(5, dtype=np.uint64) * 500
    summary = summarize_spike_band(spike_band)

    channels, spike_timestamps, amplitudes = detect_spikes(
        spike_band,
        timestamps,
        sample_rate=2000,
        summary=summary,
    )

    assert channels.tolist() == [0]
    assert spike_timestamps.tolist() == [500]
    assert amplitudes.tolist() == pytest.approx([-10.0])


def test_spike_analysis_computes_isi_violations_and_firing_rates():
    buffer = SpikeBuffer.create(capacity=8)
    buffer.add_spikes(
        np.array([0, 0, 0, 1], dtype=np.uint16),
        np.array([0, 1000, 5000, 0], dtype=np.uint64),
        np.array([-5.0, -4.5, -4.0, -3.0], dtype=np.float32),
    )

    isi = compute_isi_violations(buffer, n_channels=2)
    firing_rate = compute_firing_rate(buffer, n_channels=2, observed_duration_sec=1.0)

    assert isi.tolist() == pytest.approx([0.5, 0.0])
    assert firing_rate.tolist() == pytest.approx([3.0, 1.0])
