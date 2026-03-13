"""
Sanitization Layer for Signal Stability Gateway

Provides DSP filtering and artifact rejection for 1024-channel neural data.
"""

from .layer import SanitizationLayer

__all__ = ['SanitizationLayer']
