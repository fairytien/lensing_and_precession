"""Centralized filename builders for banks, mismatch cubes, and figures.

These helpers ensure consistent naming across the pipeline so downstream tools
can locate outputs deterministically.
"""

import os
import glob
import math
from datetime import datetime
from typing import List, Optional, Tuple
import h5py
import numpy as np


def _format_min_precision(
    value: Optional[float] = None,
    prefix: str = "",
    suffix: str = "",
    decimal_style: str = "p",
) -> str:
    """Format a number with minimal precision needed to represent it accurately.

    Examples:
        46.0 -> "46"
        46.5 -> "46p5" (default)
        46.5 with decimal_style="dot" -> "46.5"
    """
    if value is None:
        return ""

    if decimal_style not in {"p", "dot"}:
        raise ValueError("decimal_style must be 'p' or 'dot'")

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

    if decimal_style == "p":
        s = s.replace(".", "p")

    return f"{prefix}{s}{suffix}"


def _parse_decimal_token(token: str) -> float:
    """Parse numeric tokens that may use either dot or p as decimal separator."""
    return float(str(token).replace("p", "."))


def _canonical_token(value: float) -> str:
    """Return canonical numeric token used in filenames."""
    return _format_min_precision(value, decimal_style="p")


def _glob_union(patterns: List[str]) -> List[str]:
    """Return sorted unique matches across multiple glob patterns."""
    matches_set = set()
    for pattern in patterns:
        matches_set.update(glob.glob(pattern))
    return sorted(matches_set)


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
    z_name = None if z is None or _is_close(float(z), 0.0, 1e-12) else z
    name = (
        f"{prefix}{_format_min_precision(z_name, prefix='_z')}"
        f"_mcz{_format_min_precision(mcz_msun)}"
        f"_omega{_format_min_precision(omega_min)}-{_format_min_precision(omega_max)}x{omega_pts}"
        f"_theta{_format_min_precision(theta_min)}-{_format_min_precision(theta_max)}x{theta_pts}"
        f"_gamma0-2pix{gamma_pts}"
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
    omega_min: float,
    omega_max: float,
    omega_pts: int,
    theta_min: float,
    theta_max: float,
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
    z_name = None if z is None or _is_close(float(z), 0.0, 1e-12) else z
    name = (
        f"mismatch_cubes{_format_min_precision(z_name, prefix='_z')}"
        f"_mcz{_format_min_precision(mcz_msun)}"
        f"_I{_format_min_precision(I)}"
        f"_td{_format_min_precision(td_min_ms)}-{_format_min_precision(td_max_ms)}x{td_pts}"
        f"_omega{_format_min_precision(omega_min)}-{_format_min_precision(omega_max)}x{omega_pts}"
        f"_theta{_format_min_precision(theta_min)}-{_format_min_precision(theta_max)}x{theta_pts}"
        f"_gamma0-2pix{gamma_pts}"
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
    omega_min: Optional[float],
    omega_max: Optional[float],
    omega_pts: Optional[int],
    theta_min: Optional[float],
    theta_max: Optional[float],
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
    z_name = None if z is None or _is_close(float(z), 0.0, 1e-12) else z
    mcz_token = (
        f"{_format_min_precision(mcz_min)}-{_format_min_precision(mcz_max)}"
        if mcz_pts is None
        else f"{_format_min_precision(mcz_min)}-{_format_min_precision(mcz_max)}x{mcz_pts}"
    )
    td_token = (
        f"{_format_min_precision(td_min_ms)}-{_format_min_precision(td_max_ms)}"
        if td_pts is None
        else f"{_format_min_precision(td_min_ms)}-{_format_min_precision(td_max_ms)}x{td_pts}"
    )
    name = (
        f"best_match_I{_format_min_precision(I)}"
        f"{_format_min_precision(z_name, prefix='_z')}"
        f"_mcz{mcz_token}"
        f"_td{td_token}"
    )
    # Append full template-grid tokens when available.
    if (
        td_pts is not None
        and mcz_pts is not None
        and omega_min is not None
        and omega_max is not None
        and omega_pts is not None
        and theta_min is not None
        and theta_max is not None
        and theta_pts is not None
        and gamma_pts is not None
    ):
        name += (
            f"_omega{_format_min_precision(omega_min)}-{_format_min_precision(omega_max)}x{omega_pts}"
            f"_theta{_format_min_precision(theta_min)}-{_format_min_precision(theta_max)}x{theta_pts}"
            f"_gamma0-2pix{gamma_pts}"
        )

    name += f"_{orientation_tag}.h5"
    return os.path.join(best_match_dir, name)


def contour_mcz_td_filename(
    fig_dir: str,
    I: float,
    mcz_min: float,
    mcz_max: float,
    mcz_pts: Optional[int],
    td_min_ms: float,
    td_max_ms: float,
    td_pts: Optional[int],
    orientation_tag: str,
    z: Optional[float] = None,
    ext: str = "pdf",
) -> str:
    """Build the figure path for the final mismatch contour over (td, mcz).

    Returns a path under fig_dir; creates directories.
    Order: I, mcz ranges, td ranges, suffix, orientation_tag.
    """
    os.makedirs(fig_dir, exist_ok=True)
    z_name = None if z is None or _is_close(float(z), 0.0, 1e-12) else z
    name = (
        f"contour_I{_format_min_precision(I)}"
        f"{_format_min_precision(z_name, prefix='_z')}"
        f"_mcz{_format_min_precision(mcz_min)}-{_format_min_precision(mcz_max)}"
        f"{'' if mcz_pts is None else f'x{int(mcz_pts)}'}"
        f"_td{_format_min_precision(td_min_ms)}-{_format_min_precision(td_max_ms)}"
        f"{'' if td_pts is None else f'x{int(td_pts)}'}"
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


def parse_mcz_from_mismatch_cube_path(path: str) -> Optional[float]:
    """Extract the mcz value from canonical mismatch cube filenames."""
    base = os.path.basename(path)
    try:
        token = base.split("_mcz", 1)[1]
        token = token.split("_", 1)[0]
        if not token:
            return None
        return _parse_decimal_token(token)
    except Exception:
        return None


def _is_close(a: float, b: float, tol: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=float(tol))


def find_mismatch_cube_files(
    results_dir: str,
    td_min_ms: Optional[float],
    td_max_ms: Optional[float],
    orientation_tag: str,
    z: Optional[float] = None,
    mcz_min: Optional[float] = None,
    mcz_max: Optional[float] = None,
    mcz_msun: Optional[float] = None,
    mcz_tolerance: float = 1e-6,
) -> List[str]:
    """Return mismatch cube files matching the requested contour run."""
    if td_min_ms is None or td_max_ms is None:
        td_tokens = ["td*x*"]
    else:
        td_lo = _canonical_token(td_min_ms)
        td_hi = _canonical_token(td_max_ms)
        td_tokens = [f"td{td_lo}-{td_hi}x*"]

    mcz_token = "mcz*"
    if z is None:
        z_prefixes = ["mismatch_cubes_"]
    elif _is_close(float(z), 0.0, 1e-12):
        z_prefixes = ["mismatch_cubes_"]
    else:
        z_prefixes = [f"mismatch_cubes_z{_canonical_token(float(z))}_"]

    patterns = []
    for z_prefix in z_prefixes:
        for td_token in td_tokens:
            patterns.append(
                os.path.join(
                    results_dir,
                    "mismatch_cubes",
                    (
                        f"{z_prefix}{mcz_token}_I*_{td_token}"
                        f"_omega*-*x*_theta*-*x*_gamma0-2pix*_{orientation_tag}.h5"
                    ),
                )
            )
    matches = _glob_union(patterns)
    selected = []
    for path in matches:
        mcz_val = parse_mcz_from_mismatch_cube_path(path)
        if mcz_val is None:
            continue
        if mcz_msun is not None and not _is_close(mcz_val, mcz_msun, mcz_tolerance):
            continue
        if mcz_min is not None and mcz_val < mcz_min:
            continue
        if mcz_max is not None and mcz_val > mcz_max:
            continue
        selected.append(path)
    return selected


def parse_mcz_range_from_best_match_path(path: str) -> Optional[Tuple[float, float]]:
    """Extract (mcz_min, mcz_max) from canonical best-match filenames."""
    base = os.path.basename(path)
    try:
        token = base.split("_mcz", 1)[1]
        token = token.split("_", 1)[0]
        bounds = token.split("x", 1)[0]
        lo, hi = bounds.split("-", 1)
        return _parse_decimal_token(lo), _parse_decimal_token(hi)
    except Exception:
        return None


def find_best_match_file(
    results_dir: str,
    mcz_min: float,
    mcz_max: float,
    td_min_ms: float,
    td_max_ms: float,
    orientation_tag: str,
    z: Optional[float] = None,
    mcz_tolerance: float = 1e-6,
) -> Optional[str]:
    """Return the newest best-match file for the requested contour run."""
    mcz_lo = _canonical_token(mcz_min)
    mcz_hi = _canonical_token(mcz_max)
    td_lo = _canonical_token(td_min_ms)
    td_hi = _canonical_token(td_max_ms)

    if z is None:
        z_tokens = [None]
    elif _is_close(float(z), 0.0, 1e-12):
        z_tokens = [None]
    else:
        z_tokens = [_canonical_token(float(z))]

    patterns = []
    for z_tok in z_tokens:
        z_part = "" if z_tok is None else f"_z{z_tok}"
        patterns.append(
            os.path.join(
                results_dir,
                "best_match",
                (
                    f"best_match_I*{z_part}_mcz{mcz_lo}-{mcz_hi}x*_td{td_lo}-{td_hi}x*"
                    f"_omega*-*x*_theta*-*x*_gamma0-2pix*_{orientation_tag}.h5"
                ),
            )
        )
    matches = _glob_union(patterns)
    if not matches:
        return None

    selected = []
    for path in matches:
        parsed = parse_mcz_range_from_best_match_path(path)
        if parsed is None:
            continue
        lo, hi = parsed
        if not _is_close(lo, mcz_min, mcz_tolerance):
            continue
        if not _is_close(hi, mcz_max, mcz_tolerance):
            continue

        # Final authenticity check from file contents (lean and robust).
        try:
            with h5py.File(path, "r") as h5:
                if "mcz" not in h5 or "td" not in h5:
                    continue
                mcz_arr = h5["mcz"][:]
                td_arr = h5["td"][:]
                if mcz_arr.size == 0 or td_arr.size == 0:
                    continue
                if not _is_close(float(np.nanmin(mcz_arr)), mcz_min, mcz_tolerance):
                    continue
                if not _is_close(float(np.nanmax(mcz_arr)), mcz_max, mcz_tolerance):
                    continue

                td_min_s = float(td_min_ms) / 1e3
                td_max_s = float(td_max_ms) / 1e3
                if not _is_close(float(np.nanmin(td_arr)), td_min_s, mcz_tolerance):
                    continue
                if not _is_close(float(np.nanmax(td_arr)), td_max_s, mcz_tolerance):
                    continue
                if "orientation_tag" in h5.attrs:
                    if str(h5.attrs["orientation_tag"]) != str(orientation_tag):
                        continue
        except Exception:
            continue

        selected.append(path)

    if not selected:
        return None

    selected.sort(key=os.path.getmtime, reverse=True)
    return selected[0]
