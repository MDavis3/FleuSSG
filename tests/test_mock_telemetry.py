import numpy as np

from ssg.ingestion.mock_telemetry import MockTelemetry, MockTelemetryConfig


def test_mock_telemetry_generates_monotonic_batches_and_resets():
    telemetry = MockTelemetry(
        n_channels=4,
        sample_rate_hz=1_000,
        config=MockTelemetryConfig(seed=2),
    )

    samples, timestamps = telemetry.generate_batch(batch_size=8)

    assert samples.shape == (8, 4)
    assert np.all(np.diff(timestamps) > 0)
    assert telemetry.get_elapsed_time_sec() == 0.008

    telemetry.reset()

    assert telemetry.get_elapsed_time_sec() == 0.0
