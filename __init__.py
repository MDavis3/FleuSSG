"""Compatibility wrapper for the source checkout package."""

from importlib import import_module
from pathlib import Path

_inner_package = import_module(".ssg", __name__)
__path__[:] = [str(Path(__file__).resolve().parent / "ssg")]

for exported_name in getattr(_inner_package, "__all__", []):
    globals()[exported_name] = getattr(_inner_package, exported_name)

__all__ = list(getattr(_inner_package, "__all__", []))
__author__ = getattr(_inner_package, "__author__", "")
__version__ = getattr(_inner_package, "__version__", "")