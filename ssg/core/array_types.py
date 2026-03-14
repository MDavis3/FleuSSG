"""Shared NumPy type aliases for the SSG public API."""

import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float32]
BoolVector = NDArray[np.bool_]
TimestampVector = NDArray[np.uint64]
ChannelIndexVector = NDArray[np.uint16]
