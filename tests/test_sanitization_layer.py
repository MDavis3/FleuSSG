import numpy as np

from ssg.sanitization.artifacts import apply_tanh_scaling, update_artifact_state
from ssg.sanitization.filters import apply_filter_bank, design_filter_bank, init_filter_state
from ssg.sanitization.layer import SanitizationLayer


def test_sanitization_layer_processes_batches_and_resets_state():
    layer = SanitizationLayer(n_channels=4, sample_rate_hz=10_000)
    samples = np.zeros((64, 4), dtype=np.float32)
    timestamps = np.arange(64, dtype=np.uint64)

    frame = layer.sanitize_batch(samples, timestamps)

    assert frame.raw_unfiltered.shape == (64, 4)
    assert frame.lfp.shape == (64, 4)
    assert frame.spikes.shape == (64, 4)
    assert frame.artifact_flags.shape == (4,)

    layer.reset()

    np.testing.assert_array_equal(layer.get_rolling_sigma(), np.full(4, 10.0, dtype=np.float32))


def test_sanitization_layer_tanh_scaling_only_changes_flagged_channels():
    layer = SanitizationLayer(n_channels=2, sample_rate_hz=10_000)
    layer.sanitize_batch(np.ones((32, 2), dtype=np.float32), np.arange(32, dtype=np.uint64))

    data = np.array([[1000.0, 3.0], [-1000.0, -3.0]], dtype=np.float32)
    scaled = layer.apply_tanh_scaling(data, np.array([True, False]))

    assert np.all(np.abs(scaled[:, 0]) < np.abs(data[:, 0]))
    np.testing.assert_array_equal(scaled[:, 1], data[:, 1])


def test_sanitization_layer_handles_empty_timestamps_and_flags_artifacts():
    layer = SanitizationLayer(n_channels=2, sample_rate_hz=10_000)

    warmup = layer.sanitize_batch(np.zeros((32, 2), dtype=np.float32), np.array([], dtype=np.uint64))
    artifact_batch = np.zeros((32, 2), dtype=np.float32)
    artifact_batch[:, 0] = 500.0
    flagged = layer.sanitize_batch(artifact_batch, np.arange(32, dtype=np.uint64))

    assert warmup.timestamp_us == 0
    assert flagged.artifact_flags[0]
    assert not flagged.artifact_flags[1]


def test_sanitization_helpers_cover_filter_bank_and_artifact_updates():
    filter_bank = design_filter_bank(sample_rate_hz=10_000)
    filter_state = init_filter_state(filter_bank, n_channels=2)
    samples = np.ones((32, 2), dtype=np.float32)

    notched, lfp, spikes = apply_filter_bank(samples, filter_bank, filter_state)
    artifact_flags, sigma, rolling_count = update_artifact_state(
        notched,
        np.full(2, 50.0, dtype=np.float32),
        rolling_count=0,
    )

    assert notched.shape == (32, 2)
    assert lfp.shape == (32, 2)
    assert spikes.shape == (32, 2)
    assert sigma.shape == (2,)
    assert rolling_count == 32
    assert artifact_flags.shape == (2,)


def test_artifact_state_supports_transposed_sample_axis():
    data = np.array(
        [
            [0.0, 2.0, 4.0, 6.0],
            [1.0, 3.0, 5.0, 7.0],
        ],
        dtype=np.float32,
    )
    baseline_sigma = np.full(2, 50.0, dtype=np.float32)

    flags_row_major, sigma_row_major, rolling_row_major = update_artifact_state(
        data.T,
        baseline_sigma,
        rolling_count=0,
    )
    flags_col_major, sigma_col_major, rolling_col_major = update_artifact_state(
        data,
        baseline_sigma,
        rolling_count=0,
        sample_axis=1,
    )

    np.testing.assert_array_equal(flags_col_major, flags_row_major)
    np.testing.assert_allclose(sigma_col_major, sigma_row_major)
    assert rolling_col_major == rolling_row_major == 4


def test_artifact_helper_scaling_returns_unmodified_data_when_no_flags():
    data = np.array([[2.0, -2.0]], dtype=np.float32)
    scaled = apply_tanh_scaling(
        data,
        np.array([False, False]),
        np.array([1.0, 1.0], dtype=np.float32),
    )

    np.testing.assert_array_equal(scaled, data)
