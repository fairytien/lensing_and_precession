"""Shared path bootstrap helper for scripts.

Ensures the project root is available on ``sys.path`` so script-local
execution can import ``modules.*`` reliably.
"""

import os
import sys


def ensure_project_root_on_path(script_file: str) -> str:
    """Insert project root at the front of ``sys.path`` if missing.

    Args:
        script_file: ``__file__`` value from the calling script.

    Returns:
        Absolute project-root path.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(script_file)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root
