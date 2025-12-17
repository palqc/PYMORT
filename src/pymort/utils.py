"""
Backward-compatible shim forwarding to ``pymort.models.utils``.
"""

from pymort.models.utils import _estimate_rw_params, estimate_rw_params

__all__ = ["estimate_rw_params", "_estimate_rw_params"]
