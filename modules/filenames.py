"""Centralized filename builders for banks, mismatch cubes, and figures.

These helpers ensure consistent naming across the pipeline so downstream tools
can locate outputs deterministically.
"""

import os


def _format_min_precision(value: float) -> str:
    """Format a number with minimal precision needed to represent it accurately.

    Examples:
        46.0 → "46"
        46.5 → "46.5"
        46.25 → "46.25"
        46.123 → "46.123"
    """
    # Convert to string to preserve input precision
    s = str(value)

    # If it's already an integer string (no decimal point), return as-is
    if "." not in s:
        return s

    # Remove trailing zeros after decimal point, but keep at least one digit if all zeros
    s = s.rstrip("0")

    # If we stripped everything after the decimal, remove the decimal point too
    if s.endswith("."):
        s = s[:-1]

    return s


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
    prefix: str = "rp_bank",
) -> str:
    """Build the HDF5 path for a single 4D bank at fixed mcz and orientation.

    Returns a path under bank_dir; the directory is created if missing.
    """
    os.makedirs(bank_dir, exist_ok=True)
    name = (
        f"{prefix}_mcz{_format_min_precision(mcz_msun)}Msun"
        f"_omega{_format_min_precision(omega_min)}-{_format_min_precision(omega_max)}"
        f"_theta{_format_min_precision(theta_min)}-{_format_min_precision(theta_max)}"
        f"_o{omega_pts}-t{theta_pts}-g{gamma_pts}"
        f"_{orientation_tag}.h5"
    )
    return os.path.join(bank_dir, name)


def mismatch_cubes_filename(
    results_dir: str,
    mcz_msun: float,
    td_min_ms: float,
    td_max_ms: float,
    orientation_tag: str,
) -> str:
    """Build the HDF5 path for per-mcz mismatch cube outputs.

    Returns a path under results_dir/mismatch_cubes; creates directories.
    """
    mismatch_dir = os.path.join(results_dir, "mismatch_cubes")
    os.makedirs(mismatch_dir, exist_ok=True)
    name = (
        f"mismatch_cubes_mcz{_format_min_precision(mcz_msun)}Msun"
        f"_td{_format_min_precision(td_min_ms)}-{_format_min_precision(td_max_ms)}ms"
        f"_{orientation_tag}.h5"
    )
    return os.path.join(mismatch_dir, name)


def best_match_filename(
    results_dir: str,
    td_min_ms: float,
    td_max_ms: float,
    mcz_min: float,
    mcz_max: float,
    orientation_tag: str,
) -> str:
    """Build the HDF5 path for the aggregated best-match outputs across all mcz.

    Returns a path under results_dir/best_match; creates directories.
    """
    best_match_dir = os.path.join(results_dir, "best_match")
    os.makedirs(best_match_dir, exist_ok=True)
    name = (
        f"best_match_td{_format_min_precision(td_min_ms)}-{_format_min_precision(td_max_ms)}ms"
        f"_mcz{_format_min_precision(mcz_min)}-{_format_min_precision(mcz_max)}Msun_{orientation_tag}.h5"
    )
    return os.path.join(best_match_dir, name)


def contour_td_mcz_filename(
    fig_dir: str,
    td_min_ms: float,
    td_max_ms: float,
    mcz_min: float,
    mcz_max: float,
    orientation_tag: str,
    ext: str = "pdf",
) -> str:
    """Build the figure path for the final mismatch contour over (td, mcz).

    Returns a path under fig_dir; creates directories.
    """
    os.makedirs(fig_dir, exist_ok=True)
    name = (
        f"contour_td{_format_min_precision(td_min_ms)}-{_format_min_precision(td_max_ms)}ms"
        f"_mcz{_format_min_precision(mcz_min)}-{_format_min_precision(mcz_max)}Msun"
        f"_min_mismatch_{orientation_tag}.{ext}"
    )
    return os.path.join(fig_dir, name)
