"""Small argparse helpers shared by repo entry points.

Sections:
- Grid resolution helpers
- Shared argument helpers
- Grid argument groups
- Sampling and cosmology argument groups
- Chunking argument groups
"""

from argparse import ArgumentParser
from typing import Iterable, Optional

import numpy as np


# ==============================================================================
# Grid Resolution Helpers
# ==============================================================================


def resolve_grid_array(
    min_val: float,
    max_val: float,
    pts: Optional[int] = None,
    step: Optional[float] = None,
    label: Optional[str] = None,
) -> np.ndarray:
    """Return a 1-D grid from either a point-count (linspace) or step (arange-style).

    Exactly one of *pts* or *step* must be provided.
    When *step* is used the array starts at *min_val* and advances by *step*,
    stopping at or before *max_val* (numpy-arange semantics).  The generated
    array is logged so the caller can verify.
    """
    if step is not None and pts is not None:
        # step takes priority (pts is likely a default); warn if both were explicit
        pts = None
    if step is not None:
        n = int(np.floor((max_val - min_val) / step + 1e-10)) + 1
        arr = min_val + step * np.arange(n)
        if label:
            print(
                f"[step-mode] {label}: min={min_val}, max={max_val}, step={step} "
                f"-> {len(arr)} pts: {arr}"
            )
        return arr
    if pts is not None:
        return np.linspace(min_val, max_val, pts)
    raise ValueError(
        f"{'(' + label + ') ' if label else ''}Must specify either pts or step."
    )


# ==============================================================================
# Shared Argument Helpers
# ==============================================================================


def _add_grid_size_args(
    parser: ArgumentParser,
    *,
    pts_flag: str,
    default_pts: Optional[int],
    step_flag: str,
    step_help: str,
) -> ArgumentParser:
    if default_pts is not None:
        parser.add_argument(pts_flag, type=int, default=default_pts)
    parser.add_argument(step_flag, type=float, default=None, help=step_help)
    return parser


def _add_chunking_args(
    parser: ArgumentParser,
    *,
    axis_name: str,
    axis_label: str,
) -> ArgumentParser:
    parser.add_argument(
        f"--{axis_name}_chunk_index",
        type=int,
        default=None,
        help=(
            f"Chunk index for {axis_label} splitting (0-based). "
            "Defaults to SLURM_ARRAY_TASK_ID if set."
        ),
    )
    parser.add_argument(
        f"--{axis_name}_chunk_count",
        type=int,
        default=None,
        help=(
            f"Total chunks for {axis_label} splitting. "
            "Defaults to SLURM_ARRAY_TASK_COUNT if set."
        ),
    )
    return parser


# ==============================================================================
# Shared Argument Groups
# ==============================================================================


def add_orientation_args(
    parser: ArgumentParser,
    orient_choices: Optional[Iterable[str]] = None,
) -> ArgumentParser:
    """Attach the common orientation arguments used by pipeline scripts.

    Parameters
    ----------
    orient_choices : iterable of str, optional
        If provided, restricts --orient_preset to these choices.
    """
    parser.add_argument("--theta_J", type=float, default=None)
    parser.add_argument("--phi_J", type=float, default=None)
    parser.add_argument("--theta_S", type=float, default=None)
    parser.add_argument("--phi_S", type=float, default=None)
    parser.add_argument(
        "--orient_preset",
        type=str,
        default=None,
        choices=sorted(orient_choices) if orient_choices else None,
        help=(
            "Optional orientation preset to use for both params and tag."
            "If not provided, angles (theta_J, phi_J, theta_S, phi_S) form the tag."
        ),
    )
    return parser


# ==============================================================================
# Grid Argument Groups
# ==============================================================================


def add_mcz_grid_args(
    parser: ArgumentParser,
    default_min: Optional[float] = 10.0,
    default_max: Optional[float] = 90.0,
    default_pts: Optional[int] = 81,
    required: bool = False,
) -> ArgumentParser:
    """Attach mcz grid arguments used by contour pipeline scripts."""
    parser.add_argument("--mcz_min", type=float, default=default_min, required=required)
    parser.add_argument("--mcz_max", type=float, default=default_max, required=required)
    return _add_grid_size_args(
        parser,
        pts_flag="--mcz_pts",
        default_pts=default_pts,
        step_flag="--mcz_step",
        step_help="Step size for mcz grid (arange-style). Mutually exclusive with --mcz_pts.",
    )


def add_I_grid_args(
    parser: ArgumentParser,
    default_min: Optional[float] = 0.1,
    default_max: Optional[float] = 0.9,
    default_pts: Optional[int] = 41,
    required: bool = False,
) -> ArgumentParser:
    """Attach flux ratio I grid arguments used by I-td contour pipeline scripts."""
    parser.add_argument(
        "--I_min",
        type=float,
        default=default_min,
        required=required,
        help="Minimum flux ratio I (must be > 0).",
    )
    parser.add_argument(
        "--I_max",
        type=float,
        default=default_max,
        required=required,
        help="Maximum flux ratio I (must be < 1).",
    )
    return _add_grid_size_args(
        parser,
        pts_flag="--I_pts",
        default_pts=default_pts,
        step_flag="--I_step",
        step_help="Step size for I grid (arange-style). Mutually exclusive with --I_pts.",
    )


def add_td_grid_args(
    parser: ArgumentParser,
    default_min_ms: Optional[float] = 20.0,
    default_max_ms: Optional[float] = 70.0,
    default_pts: Optional[int] = 51,
    required: bool = False,
) -> ArgumentParser:
    """Attach td grid arguments used by contour pipeline scripts."""
    parser.add_argument(
        "--td_min_ms", type=float, default=default_min_ms, required=required
    )
    parser.add_argument(
        "--td_max_ms", type=float, default=default_max_ms, required=required
    )
    return _add_grid_size_args(
        parser,
        pts_flag="--td_pts",
        default_pts=default_pts,
        step_flag="--td_step_ms",
        step_help="Step size for td grid in ms (arange-style). Mutually exclusive with --td_pts.",
    )


def add_template_grid_args(
    parser: ArgumentParser,
    omega_min: float = 0.0,
    omega_max: float = 6.0,
    omega_pts: int = 61,
    theta_min: float = 0.0,
    theta_max: float = 15.0,
    theta_pts: int = 151,
    gamma_pts: int = 51,
) -> ArgumentParser:
    """Attach omega/theta/gamma grid arguments used by contour scripts."""
    parser.add_argument("--omega_min", type=float, default=omega_min)
    parser.add_argument("--omega_max", type=float, default=omega_max)
    parser.add_argument("--omega_pts", type=int, default=omega_pts)
    parser.add_argument("--theta_min", type=float, default=theta_min)
    parser.add_argument("--theta_max", type=float, default=theta_max)
    parser.add_argument("--theta_pts", type=int, default=theta_pts)
    parser.add_argument("--gamma_pts", type=int, default=gamma_pts)
    return parser


# ==============================================================================
# Sampling and Cosmology Argument Groups
# ==============================================================================


def add_frequency_args(
    parser: ArgumentParser,
    f_min: float = 20.0,
    delta_f: float = 0.25,
) -> ArgumentParser:
    """Attach common frequency sampling arguments."""
    parser.add_argument("--f_min", type=float, default=f_min)
    parser.add_argument("--delta_f", type=float, default=delta_f)
    return parser


def add_redshift_arg(
    parser: ArgumentParser,
    default_z: Optional[float] = None,
) -> ArgumentParser:
    """Attach redshift argument used by template and mismatch pipelines."""
    parser.add_argument(
        "--z",
        type=float,
        default=default_z,
        help=(
            "Source redshift. Effective detector-frame chirp mass is "
            "mcz_det = mcz * (1 + z). If omitted, no cosmology mapping is applied."
        ),
    )
    return parser


# ==============================================================================
# Chunking Argument Groups
# ==============================================================================


def add_mcz_chunking_args(parser: ArgumentParser) -> ArgumentParser:
    """Attach common mcz chunking arguments used by array jobs."""
    return _add_chunking_args(
        parser,
        axis_name="mcz",
        axis_label="mcz",
    )


def add_I_chunking_args(parser: ArgumentParser) -> ArgumentParser:
    """Attach flux ratio I chunking arguments used by array jobs."""
    return _add_chunking_args(
        parser,
        axis_name="I",
        axis_label="I",
    )
