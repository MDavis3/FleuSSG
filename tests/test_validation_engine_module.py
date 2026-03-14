import numpy as np
import pytest

from ssg.core.constants import N_CHANNELS
from ssg.core.data_types import ChannelMetrics, SanitizedFrame
from ssg.validation.engine import ValidationEngine
from ssg.validation.metrics import build_viability_mask, summarize_region, update_ema_snr


def test_validation_engine_updates_impedance_and_computes_region_metrics():
    engine = ValidationEngine(n_channels=4, sample_rate_hz=20_000)
    engine.update_impedance(np.array([500.0, 600.0, 700.0, 800.0], dtype=np.float32))

    metrics = ChannelMetrics(
        timestamp_us=0,
        snr=np.array([5.0, 4.5, 3.0, 6.0], dtype=np.float32),
        firing_rate_hz=np.array([10.0, 9.0, 8.0, 7.0], dtype=np.float32),
        isi_violation_rate=np.zeros(4, dtype=np.float32),
        impedance_kohm=np.array([500.0, 600.0, 700.0, 800.0], dtype=np.float32),
        viability_mask=np.array([True, True, False, True], dtype=bool),
        viable_channel_count=3,
    )

    region = engine.get_region_metrics(metrics, 0, 2)

    assert region.viable_count == 2
    assert region.total_count == 2
    assert region.viability_pct == 100.0


def test_validation_engine_validates_impedance_shape_and_artifact_viability():
    engine = ValidationEngine(sample_rate_hz=1_000)

    with pytest.raises(ValueError):
        engine.update_impedance(np.array([1.0, 2.0], dtype=np.float32))

    spikes = np.full((32, N_CHANNELS), 50.0, dtype=np.float32)
    frame = SanitizedFrame(
        timestamp_us=0,
        raw_unfiltered=spikes.copy(),
        lfp=spikes.copy(),
        spikes=spikes,
        artifact_flags=np.zeros(N_CHANNELS, dtype=bool),
    )
    frame.artifact_flags[0] = True
    timestamps = np.arange(32, dtype=np.uint64)

    metrics = engine.process(frame, timestamps)

    assert not metrics.viability_mask[0]
    engine.reset()
    np.testing.assert_array_equal(
        engine._ema_noise,
        np.full(N_CHANNELS, 10.0, dtype=np.float32),
    )


def test_validation_metric_helpers_update_snr_and_region_summary():
    spike_band = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ],
        dtype=np.float32,
    )

    snr, ema_noise, ema_signal = update_ema_snr(
        spike_band,
        np.ones(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        batch_count=0,
    )

    assert snr.shape == (2,)
    assert np.all(ema_noise > 0)
    assert np.all(ema_signal > 0)

    metrics = ChannelMetrics(
        timestamp_us=0,
        snr=np.array([5.0, 4.5], dtype=np.float32),
        firing_rate_hz=np.array([10.0, 9.0], dtype=np.float32),
        isi_violation_rate=np.zeros(2, dtype=np.float32),
        impedance_kohm=np.array([500.0, 600.0], dtype=np.float32),
        viability_mask=np.array([True, False], dtype=bool),
        viable_channel_count=1,
    )
    region = summarize_region(metrics, 0, 2)

    assert region.viable_count == 1
    assert region.mean_snr == pytest.approx(4.75)


def test_validation_metric_helpers_apply_viability_criteria():
    mask = build_viability_mask(
        snr=np.array([5.0, 2.0], dtype=np.float32),
        isi_violation_rate=np.array([0.0, 0.0], dtype=np.float32),
        impedance_kohm=np.array([500.0, 500.0], dtype=np.float32),
        artifact_flags=np.array([False, True]),
    )

    np.testing.assert_array_equal(mask, np.array([True, False], dtype=bool))
