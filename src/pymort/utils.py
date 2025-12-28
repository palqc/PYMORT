"""Backward-compatible shim that re-exports model utilities.

Note:
    Docstrings follow Google style for clarity and spec alignment.
"""

from pymort.models.utils import _estimate_rw_params, estimate_rw_params

__all__ = ["_estimate_rw_params", "estimate_rw_params"]
