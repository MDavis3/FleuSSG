import numpy as np

from ssg.core.ring_buffer import RingBuffer


def test_ring_buffer_returns_recent_samples_in_order():
    buffer = RingBuffer(shape=(5, 2), dtype=np.float32)
    samples = np.arange(14, dtype=np.float32).reshape(7, 2)
    timestamps = np.arange(100, 107, dtype=np.uint64)

    buffer.push_batch(samples[:3], timestamps[:3])
    buffer.push_batch(samples[3:], timestamps[3:])

    recent_samples, recent_timestamps = buffer.get_last(4)
    all_samples, all_timestamps = buffer.get_all()

    np.testing.assert_array_equal(recent_samples, samples[-4:])
    np.testing.assert_array_equal(recent_timestamps, timestamps[-4:])
    np.testing.assert_array_equal(all_samples, samples[-5:])
    np.testing.assert_array_equal(all_timestamps, timestamps[-5:])


def test_ring_buffer_clear_resets_visible_state():
    buffer = RingBuffer(shape=(3, 1), dtype=np.float32)
    buffer.push_batch(
        np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        np.array([1, 2, 3], dtype=np.uint64),
    )

    buffer.clear()

    assert buffer.current_size == 0
    assert not buffer.is_full
