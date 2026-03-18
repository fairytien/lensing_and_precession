"""Small argparse helpers shared by repo entry points."""

from argparse import ArgumentParser
from typing import Iterable, Optional


def add_orientation_args(parser: ArgumentParser) -> ArgumentParser:
    """Attach the common orientation arguments used by pipeline scripts."""
    parser.add_argument("--theta_J", type=float, default=None)
    parser.add_argument("--phi_J", type=float, default=None)
    parser.add_argument("--theta_S", type=float, default=None)
    parser.add_argument("--phi_S", type=float, default=None)
    parser.add_argument(
        "--orient_preset",
        type=str,
        default=None,
        help=(
            "Optional orientation preset to use for both params and tag."
            "If not provided, angles (theta_J, phi_J, theta_S, phi_S) form the tag."
        ),
    )
    return parser


def set_argument_choices(
    parser: ArgumentParser,
    dest: str,
    choices: Iterable[str],
) -> None:
    """Update an existing argparse action with dynamic choices."""
    for action in parser._actions:
        if getattr(action, "dest", None) == dest:
            action.choices = sorted(choices)
            return


def add_mcz_grid_args(
    parser: ArgumentParser,
    default_min: Optional[float] = 10.0,
    default_max: Optional[float] = 80.0,
    default_pts: Optional[int] = 71,
    required: bool = False,
) -> ArgumentParser:
    """Attach mcz grid arguments used by contour pipeline scripts."""
    parser.add_argument("--mcz_min", type=float, default=default_min, required=required)
    parser.add_argument("--mcz_max", type=float, default=default_max, required=required)
    if default_pts is not None:
        parser.add_argument("--mcz_pts", type=int, default=default_pts)
    return parser


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
    if default_pts is not None:
        parser.add_argument("--td_pts", type=int, default=default_pts)
    return parser


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


def add_frequency_args(
    parser: ArgumentParser,
    f_min: float = 20.0,
    delta_f: float = 0.25,
) -> ArgumentParser:
    """Attach common frequency sampling arguments."""
    parser.add_argument("--f_min", type=float, default=f_min)
    parser.add_argument("--delta_f", type=float, default=delta_f)
    return parser


def add_chunking_args(parser: ArgumentParser) -> ArgumentParser:
    """Attach common mcz chunking arguments used by array jobs."""
    parser.add_argument(
        "--mcz_chunk_index",
        type=int,
        default=None,
        help="Chunk index for mcz splitting (0-based). Defaults to SLURM_ARRAY_TASK_ID if set.",
    )
    parser.add_argument(
        "--mcz_chunk_count",
        type=int,
        default=None,
        help="Total chunks for mcz splitting. Defaults to SLURM_ARRAY_TASK_COUNT if set.",
    )
    return parser
