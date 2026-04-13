"""Compatibility wrapper for legacy imports.

The canonical production implementation now lives in modules.Classes.
"""

from modules import Classes as _impl
from modules.Classes import *  # noqa: F401,F403

__all__ = getattr(
    _impl, "__all__", [name for name in dir(_impl) if not name.startswith("_")]
)
