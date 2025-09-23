"""Centralized filename builders for banks, mismatch cubes, and figures.

These helpers ensure consistent naming across the pipeline so downstream tools
can locate outputs deterministically.
"""

import os


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
        f"{prefix}_mcz{mcz_msun:.0f}_omega{omega_min:.0f}-{omega_max:.0f}"
        f"_theta{theta_min:.0f}-{theta_max:.0f}_o{omega_pts}-t{theta_pts}-g{gamma_pts}"
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
        f"mismatch_cubes_mcz{mcz_msun:.0f}Msun_td{td_min_ms:.0f}-{td_max_ms:.0f}ms"
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
        f"best_match_td{td_min_ms:.0f}-{td_max_ms:.0f}ms"
        f"_mcz{mcz_min:.0f}-{mcz_max:.0f}Msun_{orientation_tag}.h5"
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
        f"contour_td{td_min_ms:.0f}-{td_max_ms:.0f}ms_"
        f"mcz{mcz_min:.0f}-{mcz_max:.0f}Msun_min_mismatch_{orientation_tag}.{ext}"
    )
    return os.path.join(fig_dir, name)
