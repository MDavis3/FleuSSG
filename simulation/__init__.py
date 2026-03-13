"""
Simulation Module for Signal Stability Gateway

Provides biological noise injection and end-to-end testing.
"""

from .noise_models import NoiseGenerator, ArtifactType
from .test_harness import TestHarness

__all__ = ['NoiseGenerator', 'ArtifactType', 'TestHarness']
