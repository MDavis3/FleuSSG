import numpy as np
import pytest

from ssg.ingestion.engine import IngestionEngine
from ssg.ingestion.polling import build_batch_timestamps, normalize_polled_batch


def test_ingestion_engine_polls_and_builds_timestamps_for_batch_sources():
    samples = np.arange(12, dtype=np.float32).reshape(4, 3)

    def data_source():
        return samples, 4_000

    engine = IngestionEngine(n_channels=3, sample_rate_hz=1_000, data_source=data_source)

    polled_samples, polled_timestamps = engine.poll()
    latest_samples, latest_timestamps = engine.get_latest_batch()

    np.testing.assert_array_equal(polled_samples, samples)
    np.testing.assert_array_equal(polled_timestamps, np.array([1000, 2000, 3000, 4000], dtype=np.uint64))
    np.testing.assert_array_equal(latest_samples, samples)
    np.testing.assert_array_equal(latest_timestamps, polled_timestamps)
    assert engine.total_frames == 4
    assert engine.buffer.current_size == 4
    assert int(polled_timestamps[-1]) == 4000
    assert latest_samples.shape == (4, 3)


def test_ingestion_engine_rejects_wrong_channel_shape():
    engine = IngestionEngine(n_channels=2, sample_rate_hz=1_000)

    with pytest.raises(ValueError):
        engine.ingest_frame(np.ones(3, dtype=np.float32), 1)

    assert engine.total_frames == 0
    assert engine.buffer.current_size == 0


def test_ingestion_engine_polls_single_frames_and_reports_fill_ratio():
    def data_source():
        return np.array([1.0, 2.0], dtype=np.float32), 10

    engine = IngestionEngine(
        n_channels=2,
        sample_rate_hz=1_000,
        buffer_duration_sec=0.01,
        data_source=data_source,
    )

    samples, timestamps = engine.poll()

    np.testing.assert_array_equal(samples, np.array([[1.0, 2.0]], dtype=np.float32))
    np.testing.assert_array_equal(timestamps, np.array([10], dtype=np.uint64))
    assert engine.total_frames == 1
    assert engine.buffer.current_size == 1
    assert engine.get_buffer_fill_ratio() == pytest.approx(0.1)
    assert samples.shape == (1, 2)


def test_ingestion_polling_helpers_normalize_frames_and_batches():
    batch_timestamps = build_batch_timestamps(4, sample_rate_hz=1_000, latest_timestamp_us=4_000)
    batch_samples = np.arange(12, dtype=np.float32).reshape(4, 3)
    normalized_batch, normalized_timestamps = normalize_polled_batch(
        batch_samples,
        timestamp_us=4_000,
        sample_rate_hz=1_000,
    )
    normalized_frame, frame_timestamps = normalize_polled_batch(
        np.array([1.0, 2.0], dtype=np.float32),
        timestamp_us=10,
        sample_rate_hz=1_000,
    )

    np.testing.assert_array_equal(batch_timestamps, np.array([1000, 2000, 3000, 4000], dtype=np.uint64))
    np.testing.assert_array_equal(normalized_batch, batch_samples)
    np.testing.assert_array_equal(normalized_timestamps, batch_timestamps)
    np.testing.assert_array_equal(normalized_frame, np.array([[1.0, 2.0]], dtype=np.float32))
    np.testing.assert_array_equal(frame_timestamps, np.array([10], dtype=np.uint64))
    assert normalized_batch.shape == (4, 3)
    assert normalized_frame.shape == (1, 2)
    assert int(frame_timestamps[0]) == 10
    assert int(batch_timestamps[0]) == 1000