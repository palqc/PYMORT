from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _v

try:
    __version__ = _v("pymort")
except PackageNotFoundError:
    __version__ = "0.0.dev"

__all__ = ["__version__"]
