"""Top-level package for PYMORT.

Note:
    Docstrings follow Google style to align with project standards.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _v

try:
    __version__ = _v("pymort")
except PackageNotFoundError:
    __version__ = "0.0.dev"

__all__ = ["__version__"]
