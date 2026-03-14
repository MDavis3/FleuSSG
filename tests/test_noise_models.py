import numpy as np

from ssg.simulation.noise_models import ArtifactInjectionConfig, ArtifactType, NoiseGenerator


def test_artifact_type_members_remain_stable():
    assert [member.name for member in ArtifactType] == [
        "JAW_CLENCH",
        "ELECTRODE_DRIFT",
        "MOTION_SPIKE",
    ]


def test_noise_generator_uses_config_object_and_tracks_typed_history():
    generator = NoiseGenerator(n_channels=16, sample_rate_hz=1000, seed=3)
    samples = np.zeros((200, 16), dtype=np.float32)

    _, artifacts = generator.inject_artifacts(
        samples,
        config=ArtifactInjectionConfig(
            jaw_clench_prob=20.0,
            electrode_drift_prob=10.0,
            motion_spike_prob=5.0,
            baseline_noise_uv=2.0,
        ),
    )

    assert artifacts == generator.get_artifact_history()
