"""Centralized filename builders for banks, mismatch cubes, and figures.

These helpers ensure consistent naming across the pipeline so downstream tools
can locate outputs deterministically.
"""

import os
from datetime import datetime
from typing import Optional, Tuple
import h5py


def _format_min_precision(
    value: Optional[float] = None, prefix: str = "", suffix: str = ""
) -> str:
    """Format a number with minimal precision needed to represent it accurately.

    Examples:
        46.0 → "46"
        46.5 → "46.5"
        46.25 → "46.25"
        46.123 → "46.123"
    """
    if value is None:
        return ""

    # Convert to string to preserve input precision
    s = str(value)

    # If it's already an integer string (no decimal point), return as-is
    if "." not in s:
        return f"{prefix}{s}{suffix}"

    # Remove trailing zeros after decimal point, but keep at least one digit if all zeros
    s = s.rstrip("0")

    # If we stripped everything after the decimal, remove the decimal point too
    if s.endswith("."):
        s = s[:-1]

    return f"{prefix}{s}{suffix}"


def timestamp_path(
    path: Optional[str], dt: Optional[datetime] = None, prefix: str = "_"
) -> Optional[str]:
    """Append date-time tag to file path stem, preserving extension.

    Uses local current time when dt is not provided.

    Examples:
        "fig.pdf", datetime(2026, 3, 3, 14, 5, 9) -> "fig_20260303_140509.pdf"
        None, datetime(...) -> None
        "fig.pdf", None -> "fig_YYYYMMDD_HHMMSS.pdf" (current local time)
    """
    if path is None:
        return None

    timestamp = (dt or datetime.now()).strftime("%Y%m%d_%H%M%S")
    root, ext = os.path.splitext(path)
    return f"{root}{prefix}{timestamp}{ext}"


def bank_filename(
    bank_dir: str,
    mcz_msun: float,
    omega_min: float,
    omega_max: float,
    omega_pts: int,
    theta_min: float,
    theta_max: float,
    theta_pts: int,
    gamma_pts: int,
    orientation_tag: str,
    z: Optional[float] = None,
    prefix: str = "rp_bank",
) -> str:
    """Build the HDF5 path for a single 4D bank at fixed mcz and orientation.

    Returns a path under bank_dir; the directory is created if missing.
    """
    os.makedirs(bank_dir, exist_ok=True)
    name = (
        f"{prefix}{_format_min_precision(z, prefix='_z')}"
        f"_mcz{_format_min_precision(mcz_msun, suffix='Msun')}"
        f"_omega{_format_min_precision(omega_min)}-{_format_min_precision(omega_max)}"
        f"_theta{_format_min_precision(theta_min)}-{_format_min_precision(theta_max)}"
        f"_o{omega_pts}-t{theta_pts}-g{gamma_pts}"
        f"_{orientation_tag}.h5"
    )
    return os.path.join(bank_dir, name)


def mismatch_cube_filename(
    results_dir: str,
    mcz_msun: float,
    I: float,
    td_min_ms: float,
    td_max_ms: float,
    td_pts: int,
    omega_pts: int,
    theta_pts: int,
    gamma_pts: int,
    orientation_tag: str,
    z: Optional[float] = None,
) -> str:
    """Build the HDF5 path for per-mcz mismatch cube outputs.

    Returns a path under results_dir/mismatch_cubes; creates directories.
    Order: mcz, I, td ranges, td-o-t-g resolution, orientation_tag.
    """
    mismatch_dir = os.path.join(results_dir, "mismatch_cubes")
    os.makedirs(mismatch_dir, exist_ok=True)
    name = (
        f"mismatch_cubes{_format_min_precision(z, prefix='_z')}"
        f"_mcz{_format_min_precision(mcz_msun, suffix='Msun')}"
        f"_I{_format_min_precision(I)}"
        f"_td{_format_min_precision(td_min_ms)}-{_format_min_precision(td_max_ms, suffix='ms')}"
        f"_td{td_pts}-o{omega_pts}-t{theta_pts}-g{gamma_pts}"
        f"_{orientation_tag}.h5"
    )
    return os.path.join(mismatch_dir, name)


def best_match_mcz_td_filename(
    results_dir: str,
    I: float,
    mcz_min: float,
    mcz_max: float,
    mcz_pts: Optional[int],
    td_min_ms: float,
    td_max_ms: float,
    td_pts: Optional[int],
    omega_pts: Optional[int],
    theta_pts: Optional[int],
    gamma_pts: Optional[int],
    orientation_tag: str,
    z: Optional[float] = None,
) -> str:
    """Build the HDF5 path for the aggregated best-match outputs across all mcz.

    Returns a path under results_dir/best_match; creates directories.
    Order: I, mcz ranges, td ranges, mcz-td-o-t-g resolution, orientation_tag.
    """
    best_match_dir = os.path.join(results_dir, "best_match")
    os.makedirs(best_match_dir, exist_ok=True)
    name = (
        f"best_match_I{_format_min_precision(I)}"
        f"{_format_min_precision(z, prefix='_z')}"
        f"_mcz{_format_min_precision(mcz_min)}-{_format_min_precision(mcz_max, suffix='Msun')}"
        f"_td{_format_min_precision(td_min_ms)}-{_format_min_precision(td_max_ms, suffix='ms')}"
    )
    # Append resolution suffix in m-td-o-t-g order if all are present
    if (
        td_pts is not None
        and mcz_pts is not None
        and omega_pts is not None
        and theta_pts is not None
        and gamma_pts is not None
    ):
        name += f"_m{mcz_pts}-td{td_pts}-o{omega_pts}-t{theta_pts}-g{gamma_pts}"

    name += f"_{orientation_tag}.h5"
    return os.path.join(best_match_dir, name)


def contour_mcz_td_filename(
    fig_dir: str,
    I: float,
    mcz_min: float,
    mcz_max: float,
    td_min_ms: float,
    td_max_ms: float,
    orientation_tag: str,
    z: Optional[float] = None,
    ext: str = "pdf",
) -> str:
    """Build the figure path for the final mismatch contour over (td, mcz).

    Returns a path under fig_dir; creates directories.
    Order: I, mcz ranges, td ranges, suffix, orientation_tag.
    """
    os.makedirs(fig_dir, exist_ok=True)
    name = (
        f"contour_I{_format_min_precision(I)}"
        f"{_format_min_precision(z, prefix='_z')}"
        f"_mcz{_format_min_precision(mcz_min)}-{_format_min_precision(mcz_max, suffix='Msun')}"
        f"_td{_format_min_precision(td_min_ms)}-{_format_min_precision(td_max_ms, suffix='ms')}"
        f"_min_mismatch_{orientation_tag}.{ext}"
    )
    return os.path.join(fig_dir, name)


def get_mismatch_cube_resolution(h5_file) -> Tuple[int, int, int, int]:
    """Infer (td_pts, omega_pts, theta_pts, gamma_pts) from an opened HDF5 cube.
    Expects explicit axis datasets: 'td', 'omega', 'theta', 'gamma'.
    """
    td_pts = int(h5_file["td"].shape[0])
    omega_pts = int(h5_file["omega"].shape[0])
    theta_pts = int(h5_file["theta"].shape[0])
    gamma_pts = int(h5_file["gamma"].shape[0])
    return td_pts, omega_pts, theta_pts, gamma_pts
