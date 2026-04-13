"""Compatibility wrapper for legacy imports.

The canonical production implementation now lives in modules.default_params.
"""

from modules import default_params as _impl
from modules.default_params import *  # noqa: F401,F403

__all__ = getattr(
    _impl, "__all__", [name for name in dir(_impl) if not name.startswith("_")]
)
