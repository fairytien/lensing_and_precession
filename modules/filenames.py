"""Centralized filename builders for banks, mismatch cubes, and figures.

Sections:
- Shared path and token helpers
- Run directory helpers
- Shared artifact filename helpers
- Shared inspection and parsing helpers
- `mcz_td` naming helpers
- `mcz_td` inspection and discovery helpers
- `I_td` naming helpers
- `I_td` inspection and discovery helpers

Provides:
- Run-directory builders for template-bank, `mcz_td`, and `I_td` workflows
- Canonical filename builders for bank, mismatch cube, best-match, and contour outputs
- Parsing and discovery helpers for canonical pipeline filenames

Design goals: consistent naming, shared token grammar, and behavior-preserving
refactors that keep downstream filename discovery stable.
"""

import glob
import math
import os
import re
from datetime import datetime
from typing import List, Optional, Tuple, cast

import h5py
import numpy as np

# Omitted redshift is encoded explicitly via this token.
_Z_NONE_TOKEN = "NaN"
_DEFAULT_SHARED_ROOT = "/work/10000/fairytien33/gw_shared_data"

_TEMPLATE_GRID_TOKEN_RE = re.compile(
    r"_omega([^_]+)-([^x_]+)x(\d+)_theta([^_]+)-([^x_]+)x(\d+)_gamma0-2pix(\d+)_"
)


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

    s = format(float(value), ".15g")

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


def _z_dir_token(z: Optional[float]) -> str:
    """Return a stable z token for directory names."""
    return f"z{_canonical_z_token(z)}"


def _canonical_z_token(z: Optional[float]) -> str:
    """Return canonical z token used in filenames/directories."""
    if z is None:
        return _Z_NONE_TOKEN
    z_val = float(z)
    if np.isnan(z_val):
        return _Z_NONE_TOKEN
    return _canonical_token(z_val)


def _with_optional_orientation_suffix(path: str, orientation_tag: Optional[str]) -> str:
    """Append orientation suffix once when provided."""
    if orientation_tag is None:
        return path
    tag = str(orientation_tag).strip()
    if not tag:
        return path

    # Avoid producing ".../_tag" for inputs that include trailing separators.
    trimmed_path = path.rstrip("/\\")
    if not trimmed_path:
        return path
    suffix = f"_{tag}"
    if os.path.basename(trimmed_path).endswith(suffix):
        return trimmed_path
    return f"{trimmed_path}{suffix}"


# ==============================================================================
# Shared Path And Token Helpers
# ==============================================================================


def _normalize_dir_path(path: str) -> str:
    """Trim trailing separators while preserving the original fallback."""
    return path.rstrip("/\\") or path


def _ensure_dir(path: str) -> str:
    """Create a directory when needed and return it for inline use."""
    os.makedirs(path, exist_ok=True)
    return path


def default_shared_data_root() -> str:
    """Return the shared cluster data root, overridable via env var."""
    return _normalize_dir_path(os.environ.get("SHARED_DATA_ROOT", _DEFAULT_SHARED_ROOT))


def default_template_bank_base_dir() -> str:
    """Return the shared base directory used for template-bank runs."""
    return os.path.join(default_shared_data_root(), "template_banks")


def default_mismatch_base_dir() -> str:
    """Return the shared base directory used for mismatch runs."""
    return os.path.join(default_shared_data_root(), "mismatch")


def _range_token(
    lower: float,
    upper: float,
    pts: Optional[int] = None,
    *,
    coerce_int: bool = False,
) -> str:
    """Build a canonical lower-upper token with an optional point count."""
    token = f"{_format_min_precision(lower)}-{_format_min_precision(upper)}"
    if pts is not None:
        pts_token = int(pts) if coerce_int else pts
        token += f"x{pts_token}"
    return token


def _template_grid_token(
    omega_min: float,
    omega_max: float,
    omega_pts: int,
    theta_min: float,
    theta_max: float,
    theta_pts: int,
    gamma_pts: int,
) -> str:
    """Build the shared omega/theta/gamma grid token used across artifacts."""
    return (
        f"omega{_range_token(omega_min, omega_max, omega_pts)}"
        f"_theta{_range_token(theta_min, theta_max, theta_pts)}"
        f"_gamma0-2pix{gamma_pts}"
    )


def _build_named_path(
    directory: str,
    name_parts: List[str],
    orientation_tag: str,
    *,
    ext: str = "h5",
) -> str:
    """Join canonical name parts, append orientation, and return full path."""
    stem = "_".join(part for part in name_parts if part)
    return os.path.join(_ensure_dir(directory), f"{stem}_{orientation_tag}.{ext}")


def _basename_has_all_tokens(path: str, tokens: Tuple[str, ...]) -> bool:
    """Return whether the basename already carries every required token."""
    base_name = os.path.basename(path)
    return all(token in base_name for token in tokens)


def _build_run_dir(
    base_dir: str,
    required_tokens: Tuple[str, ...],
    parts: List[str],
    orientation_tag: Optional[str],
) -> str:
    """Build a tagged run directory unless the basename is already canonical."""
    normalized_base_dir = _normalize_dir_path(base_dir)
    run_dir = normalized_base_dir
    if not _basename_has_all_tokens(normalized_base_dir, required_tokens):
        run_dir = "_".join([normalized_base_dir, *parts])
    return _with_optional_orientation_suffix(run_dir, orientation_tag)


def _token_after_marker(base: str, marker: str) -> Optional[str]:
    """Extract the token that immediately follows a filename marker."""
    try:
        token = base.split(marker, 1)[1]
        token = token.split("_", 1)[0]
    except Exception:
        return None
    return token or None


def _parse_scalar_token_from_path(path: str, marker: str) -> Optional[float]:
    """Parse a single numeric token that follows a filename marker."""
    token = _token_after_marker(os.path.basename(path), marker)
    if token is None:
        return None
    try:
        return _parse_decimal_token(token)
    except Exception:
        return None


def _parse_range_token_from_path(
    path: str, marker: str
) -> Optional[Tuple[float, float]]:
    """Parse a lower-upper numeric token that follows a filename marker."""
    token = _token_after_marker(os.path.basename(path), marker)
    if token is None:
        return None
    try:
        bounds = token.split("x", 1)[0]
        lo, hi = bounds.split("-", 1)
        return _parse_decimal_token(lo), _parse_decimal_token(hi)
    except Exception:
        return None


def _td_glob_tokens(
    td_min_ms: Optional[float], td_max_ms: Optional[float]
) -> List[str]:
    """Return canonical td glob tokens for exact or wildcard searches."""
    if td_min_ms is None or td_max_ms is None:
        return ["td*x*"]
    return [f"td{_range_token(td_min_ms, td_max_ms)}x*"]


def _dataset_bounds_match(
    values: np.ndarray,
    expected_min: float,
    expected_max: float,
    tolerance: float,
) -> bool:
    """Return whether a dataset spans the expected inclusive bounds."""
    if values.size == 0:
        return False
    return _is_close(float(np.nanmin(values)), expected_min, tolerance) and _is_close(
        float(np.nanmax(values)), expected_max, tolerance
    )


def _read_h5_array(h5: h5py.File, dataset_name: str) -> np.ndarray:
    """Read a named HDF5 dataset as a NumPy array."""
    obj = h5[dataset_name]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"Expected '{dataset_name}' to be an HDF5 dataset.")
    return np.asarray(cast(h5py.Dataset, obj)[:])


def _best_match_file_matches(
    path: str,
    sweep_dataset: str,
    sweep_min: float,
    sweep_max: float,
    td_min_ms: float,
    td_max_ms: float,
    orientation_tag: str,
    tolerance: float,
) -> bool:
    """Verify that a best-match file matches the requested sweep bounds."""
    try:
        with h5py.File(path, "r") as h5:
            if sweep_dataset not in h5 or "td" not in h5:
                return False
            if not _dataset_bounds_match(
                _read_h5_array(h5, sweep_dataset), sweep_min, sweep_max, tolerance
            ):
                return False

            td_min_s = float(td_min_ms) / 1e3
            td_max_s = float(td_max_ms) / 1e3
            if not _dataset_bounds_match(
                _read_h5_array(h5, "td"), td_min_s, td_max_s, tolerance
            ):
                return False
            if "orientation_tag" in h5.attrs:
                if str(h5.attrs["orientation_tag"]) != str(orientation_tag):
                    return False
    except Exception:
        return False

    return True


# ==============================================================================
# Run Directory Helpers
# ==============================================================================


def template_bank_run_dir(
    base_dir: str, z: Optional[float], orientation_tag: Optional[str] = None
) -> str:
    """Return bank output directory with explicit redshift token.

    Appends orientation tag suffix when provided.

    If base_dir already ends with a z token (e.g., _z0p2), it is returned unchanged.
    """
    base_dir = _normalize_dir_path(base_dir)
    run_dir = base_dir
    base_name = os.path.basename(base_dir)
    if not (base_name.startswith("template_banks_") and "_z" in base_name):
        run_dir = f"{base_dir}_{_z_dir_token(z)}"
    return _with_optional_orientation_suffix(run_dir, orientation_tag)


def contour_run_dir(
    base_dir: str,
    I: float,
    mcz_min: float,
    mcz_max: float,
    td_min_ms: float,
    td_max_ms: float,
    z: Optional[float],
    orientation_tag: Optional[str] = None,
) -> str:
    """Return run directory tagged by I, z, mcz/td ranges.

    Appends orientation tag suffix when provided.

    If base_dir already appears to include I/z/mcz/td tokens, it is returned unchanged.
    """
    return _build_run_dir(
        base_dir,
        ("_I", "_z", "_mcz", "_td"),
        [
            f"I{_canonical_token(float(I))}",
            _z_dir_token(z),
            f"mcz{_range_token(float(mcz_min), float(mcz_max))}",
            f"td{_range_token(float(td_min_ms), float(td_max_ms))}",
        ],
        orientation_tag,
    )


# ==============================================================================
# Shared Artifact Filename Helpers
# ==============================================================================


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
    z_token = _canonical_z_token(z)
    return _build_named_path(
        bank_dir,
        [
            prefix,
            f"z{z_token}",
            f"mcz{_format_min_precision(mcz_msun)}",
            _template_grid_token(
                omega_min,
                omega_max,
                omega_pts,
                theta_min,
                theta_max,
                theta_pts,
                gamma_pts,
            ),
        ],
        orientation_tag,
    )


# ==============================================================================
# Shared Inspection And Parsing Helpers
# ==============================================================================


def get_mismatch_cube_resolution(h5_file) -> Tuple[int, int, int, int]:
    """Infer (td_pts, omega_pts, theta_pts, gamma_pts) from an opened HDF5 cube.
    Expects explicit axis datasets: 'td', 'omega', 'theta', 'gamma'.
    """
    td_pts = int(h5_file["td"].shape[0])
    omega_pts = int(h5_file["omega"].shape[0])
    theta_pts = int(h5_file["theta"].shape[0])
    gamma_pts = int(h5_file["gamma"].shape[0])
    return td_pts, omega_pts, theta_pts, gamma_pts


def _is_close(a: float, b: float, tol: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=float(tol))


def parse_template_grid_tokens(path: str) -> Optional[dict]:
    """Extract omega/theta/gamma grid parameters from a canonical filename.

    Returns a dict with keys omega_min, omega_max, omega_pts, theta_min,
    theta_max, theta_pts, gamma_pts, or None if the pattern is not found.
    """
    base = os.path.basename(path)
    match = _TEMPLATE_GRID_TOKEN_RE.search(base)
    if not match:
        return None
    return {
        "omega_min": _parse_decimal_token(match.group(1)),
        "omega_max": _parse_decimal_token(match.group(2)),
        "omega_pts": int(match.group(3)),
        "theta_min": _parse_decimal_token(match.group(4)),
        "theta_max": _parse_decimal_token(match.group(5)),
        "theta_pts": int(match.group(6)),
        "gamma_pts": int(match.group(7)),
    }


# ==============================================================================
# (td, mcz) Naming Helpers
# ==============================================================================


def mismatch_mcz_cube_filename(
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
    Order: mismatch family, z, mcz, I, td range, grid resolution, orientation tag.
    """
    z_token = _canonical_z_token(z)
    return _build_named_path(
        os.path.join(results_dir, "mismatch_cubes"),
        [
            "mismatch_cubes",
            f"z{z_token}",
            f"mcz{_format_min_precision(mcz_msun)}",
            f"I{_format_min_precision(I)}",
            f"td{_range_token(td_min_ms, td_max_ms, td_pts)}",
            _template_grid_token(
                omega_min,
                omega_max,
                omega_pts,
                theta_min,
                theta_max,
                theta_pts,
                gamma_pts,
            ),
        ],
        orientation_tag,
    )


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
    Order: best_match family, I, z, mcz range, td range, optional grid, orientation tag.
    """
    z_token = _canonical_z_token(z)
    name_parts = [
        "best_match",
        f"I{_format_min_precision(I)}",
        f"z{z_token}",
        f"mcz{_range_token(mcz_min, mcz_max, mcz_pts)}",
        f"td{_range_token(td_min_ms, td_max_ms, td_pts)}",
    ]
    # Append full template-grid tokens when available.
    grid_params = (
        omega_min,
        omega_max,
        omega_pts,
        theta_min,
        theta_max,
        theta_pts,
        gamma_pts,
    )
    if td_pts is not None and mcz_pts is not None and None not in grid_params:
        name_parts.append(
            _template_grid_token(
                *cast(Tuple[float, float, int, float, float, int, int], grid_params)
            )
        )
    return _build_named_path(
        os.path.join(results_dir, "best_match"), name_parts, orientation_tag
    )


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
    Order: contour family, I, z, mcz range, td range, product suffix, orientation tag.
    """
    z_token = _canonical_z_token(z)
    return _build_named_path(
        fig_dir,
        [
            "contour",
            f"I{_format_min_precision(I)}",
            f"z{z_token}",
            f"mcz{_range_token(mcz_min, mcz_max, mcz_pts, coerce_int=True)}",
            f"td{_range_token(td_min_ms, td_max_ms, td_pts, coerce_int=True)}",
            "min_mismatch",
        ],
        orientation_tag,
        ext=ext,
    )


def compare_mcz_td_figure_filename(
    fig_dir: str,
    I: float,
    orientation_tags: List[str],
    z: Optional[float] = None,
    ext: str = "pdf",
) -> str:
    """Build the figure path for compare_Lensing outputs over (td, mcz).

    Encodes the fixed flux ratio, redshift, and orientation tags of the
    precessing panels.  Extrema-overlay style is intentionally not encoded.
    """
    z_token = _canonical_z_token(z)
    tags_token = "_".join(str(t).strip() for t in orientation_tags)
    stem = "_".join(
        [
            "compare_Lensing",
            f"I{_format_min_precision(I)}",
            f"z{z_token}",
            tags_token,
        ]
    )
    return os.path.join(_ensure_dir(fig_dir), f"{stem}.{ext}")


def parse_mcz_td_run_dir_metadata(path: str) -> Optional[dict]:
    """Extract canonical mcz_td run metadata from a run or mismatch_cubes path.

    Accepts either the tagged run directory itself or its mismatch_cubes/
    child directory.
    """
    normalized_path = _normalize_dir_path(path)
    base_name = os.path.basename(normalized_path)
    if base_name == "mismatch_cubes":
        base_name = os.path.basename(os.path.dirname(normalized_path))

    I_val = _parse_scalar_token_from_path(base_name, "_I")
    z_val = _parse_scalar_token_from_path(base_name, "_z")
    mcz_range = _parse_range_token_from_path(base_name, "_mcz")
    td_range = _parse_range_token_from_path(base_name, "_td")
    orient_match = re.search(r"_([A-Za-z0-9]+_[A-Za-z0-9]+)$", base_name)

    if (
        I_val is None
        or z_val is None
        or mcz_range is None
        or td_range is None
        or orient_match is None
    ):
        return None

    return {
        "I": I_val,
        "z": z_val,
        "mcz_min": mcz_range[0],
        "mcz_max": mcz_range[1],
        "td_min_ms": td_range[0],
        "td_max_ms": td_range[1],
        "orientation_tag": orient_match.group(1),
    }


def mismatch_sweep_mcz_td_filename(
    fig_dir: str,
    I: float,
    td_ms: float,
    mcz_min: float,
    mcz_max: float,
    omega_min: float,
    omega_max: float,
    omega_pts: int,
    theta_min: float,
    theta_max: float,
    theta_pts: int,
    gamma_pts: int,
    orientation_tag: str,
    z: Optional[float] = None,
    ext: str = "gif",
) -> str:
    """Build the derived mismatch-sweep visualization path for fixed-td mcz sweeps.

    Returns a path under fig_dir; creates directories.
    Order: mismatch_sweep family, I, td slice, z, mcz range, template grid,
    orientation tag.
    """
    z_token = _canonical_z_token(z)
    return _build_named_path(
        fig_dir,
        [
            "mismatch_sweep",
            f"I{_format_min_precision(I)}",
            f"td{_format_min_precision(td_ms)}",
            f"z{z_token}",
            f"mcz{_range_token(mcz_min, mcz_max)}",
            _template_grid_token(
                omega_min,
                omega_max,
                omega_pts,
                theta_min,
                theta_max,
                theta_pts,
                gamma_pts,
            ),
        ],
        orientation_tag,
        ext=ext,
    )


# ==============================================================================
# (td, mcz) Inspection And Discovery Helpers
# ==============================================================================


def parse_mcz_from_mismatch_mcz_cube_path(path: str) -> Optional[float]:
    """Extract the mcz value from canonical mismatch cube filenames."""
    return _parse_scalar_token_from_path(path, "_mcz")


def find_mismatch_mcz_cube_files(
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
    td_tokens = _td_glob_tokens(td_min_ms, td_max_ms)
    z_token = _canonical_z_token(z)

    patterns = [
        os.path.join(
            results_dir,
            "mismatch_cubes",
            (
                f"mismatch_cubes_z{z_token}_mcz*_I*_{td_token}"
                f"_omega*-*x*_theta*-*x*_gamma0-2pix*_{orientation_tag}.h5"
            ),
        )
        for td_token in td_tokens
    ]
    matches = _glob_union(patterns)
    selected = []
    for path in matches:
        mcz_val = parse_mcz_from_mismatch_mcz_cube_path(path)
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


def parse_mcz_range_from_best_match_mcz_td_path(
    path: str,
) -> Optional[Tuple[float, float]]:
    """Extract (mcz_min, mcz_max) from canonical best-match filenames."""
    return _parse_range_token_from_path(path, "_mcz")


def find_best_match_mcz_td_file(
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
    z_token = _canonical_z_token(z)
    pattern = os.path.join(
        results_dir,
        "best_match",
        (
            f"best_match_I*_z{z_token}_mcz{_range_token(mcz_min, mcz_max)}x*"
            f"_td{_range_token(td_min_ms, td_max_ms)}x*"
            f"_omega*-*x*_theta*-*x*_gamma0-2pix*_{orientation_tag}.h5"
        ),
    )
    matches = _glob_union([pattern])
    if not matches:
        return None

    selected = []
    for path in matches:
        parsed = parse_mcz_range_from_best_match_mcz_td_path(path)
        if parsed is None:
            continue
        lo, hi = parsed
        if not _is_close(lo, mcz_min, mcz_tolerance):
            continue
        if not _is_close(hi, mcz_max, mcz_tolerance):
            continue

        if not _best_match_file_matches(
            path,
            "mcz",
            mcz_min,
            mcz_max,
            td_min_ms,
            td_max_ms,
            orientation_tag,
            mcz_tolerance,
        ):
            continue

        selected.append(path)

    if not selected:
        return None

    selected.sort(key=os.path.getmtime, reverse=True)
    return selected[0]


# ==============================================================================
# (td, I) Naming Helpers
# ==============================================================================


def contour_I_td_run_dir(
    base_dir: str,
    mcz: float,
    I_min: float,
    I_max: float,
    td_min_ms: float,
    td_max_ms: float,
    z: Optional[float],
    orientation_tag: Optional[str] = None,
) -> str:
    """Return run directory tagged by z, mcz, I range, td range for I-td pipeline.

    Appends orientation tag suffix when provided.

    If base_dir already appears to include z/mcz/I/td tokens, it is returned unchanged.
    """
    return _build_run_dir(
        base_dir,
        ("_z", "_mcz", "_I", "_td"),
        [
            _z_dir_token(z),
            f"mcz{_canonical_token(float(mcz))}",
            f"I{_range_token(float(I_min), float(I_max))}",
            f"td{_range_token(float(td_min_ms), float(td_max_ms))}",
        ],
        orientation_tag,
    )


def mismatch_I_cube_filename(
    results_dir: str,
    I: float,
    mcz_msun: float,
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
    """Build the HDF5 path for per-I mismatch cube outputs (I-td pipeline).

    Returns a path under results_dir/mismatch_cubes; creates directories.
    Order: mismatch family, z, mcz, I, td range, grid resolution, orientation tag.
    """
    z_token = _canonical_z_token(z)
    return _build_named_path(
        os.path.join(results_dir, "mismatch_cubes"),
        [
            "mismatch_cubes",
            f"z{z_token}",
            f"mcz{_format_min_precision(mcz_msun)}",
            f"I{_format_min_precision(I)}",
            f"td{_range_token(td_min_ms, td_max_ms, td_pts)}",
            _template_grid_token(
                omega_min,
                omega_max,
                omega_pts,
                theta_min,
                theta_max,
                theta_pts,
                gamma_pts,
            ),
        ],
        orientation_tag,
    )


def parse_I_from_mismatch_I_cube_path(path: str) -> Optional[float]:
    """Extract the I value from canonical mismatch I-cube filenames."""
    return _parse_scalar_token_from_path(path, "_I")


def best_match_I_td_filename(
    results_dir: str,
    mcz_msun: float,
    I_min: float,
    I_max: float,
    I_pts: Optional[int],
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
    file_prefix: str = "best_match",
) -> str:
    """Build the HDF5 path for aggregated best-match outputs across all I (I-td pipeline).

    Returns a path under results_dir/best_match; creates directories.
    Order: aggregate prefix, z, mcz, I range, td range, optional grid, orientation tag.
    """
    prefix = str(file_prefix).strip()
    if not prefix:
        raise ValueError("file_prefix must be a non-empty string.")

    z_token = _canonical_z_token(z)
    name_parts = [
        prefix,
        f"z{z_token}",
        f"mcz{_format_min_precision(mcz_msun)}",
        f"I{_range_token(I_min, I_max, I_pts)}",
        f"td{_range_token(td_min_ms, td_max_ms, td_pts)}",
    ]
    # Append full template-grid tokens when available.
    grid_params = (
        omega_min,
        omega_max,
        omega_pts,
        theta_min,
        theta_max,
        theta_pts,
        gamma_pts,
    )
    if td_pts is not None and I_pts is not None and None not in grid_params:
        name_parts.append(
            _template_grid_token(
                *cast(Tuple[float, float, int, float, float, int, int], grid_params)
            )
        )
    return _build_named_path(
        os.path.join(results_dir, "best_match"), name_parts, orientation_tag
    )


def contour_I_td_filename(
    fig_dir: str,
    mcz_msun: float,
    I_min: float,
    I_max: float,
    I_pts: Optional[int],
    td_min_ms: float,
    td_max_ms: float,
    td_pts: Optional[int],
    orientation_tag: str,
    z: Optional[float] = None,
    ext: str = "pdf",
) -> str:
    """Build the figure path for the final mismatch contour over (td, I).

    Returns a path under fig_dir; creates directories.
    Order: contour family, z, mcz, I range, td range, product suffix, orientation tag.
    """
    z_token = _canonical_z_token(z)
    return _build_named_path(
        fig_dir,
        [
            "contour",
            f"z{z_token}",
            f"mcz{_format_min_precision(mcz_msun)}",
            f"I{_range_token(I_min, I_max, I_pts, coerce_int=True)}",
            f"td{_range_token(td_min_ms, td_max_ms, td_pts, coerce_int=True)}",
            "min_mismatch",
        ],
        orientation_tag,
        ext=ext,
    )


def compare_I_td_figure_filename(
    fig_dir: str,
    template_family: str,
    mcz_values: List[float],
    orientation_tag: str,
    z: Optional[float] = None,
    ext: str = "pdf",
) -> str:
    """Build the figure path for compare_Lensingvs<family> outputs over (td, I).

    The orientation tag is inserted before the z/mcz parameter tokens.
    Extrema-overlay style is intentionally not encoded in the filename.
    """
    orientation_token = str(orientation_tag).strip() or "orientation"
    z_token = _canonical_z_token(z)
    mcz_token = "_".join(_format_min_precision(float(value)) for value in mcz_values)
    stem = "_".join(
        [
            f"compare_Lensingvs{str(template_family).strip().upper()}",
            orientation_token,
            f"z{z_token}",
            f"mcz{mcz_token}",
        ]
    )
    return os.path.join(_ensure_dir(fig_dir), f"{stem}.{ext}")


def compare_systems_I_td_figure_filename(
    fig_dir: str,
    template_family: str,
    mcz_msun: float,
    orientation_tags: List[str],
    z: Optional[float] = None,
    ext: str = "pdf",
) -> str:
    """Build the figure path for compare_systems outputs over (td, I).

    Encodes the template family, fixed chirp mass, redshift, and the
    orientation tags of the panels.
    """
    z_token = _canonical_z_token(z)
    tags_token = "_".join(str(t).strip() for t in orientation_tags)
    stem = "_".join(
        [
            f"compare_Lensingvs{str(template_family).strip().upper()}",
            f"z{z_token}",
            f"mcz{_format_min_precision(float(mcz_msun))}",
            tags_token,
        ]
    )
    return os.path.join(_ensure_dir(fig_dir), f"{stem}.{ext}")


def bestfit_prec_params_I_td_figure_filename(
    fig_dir: str,
    mcz_values: List[float],
    I_min: float,
    I_max: float,
    td_min_ms: float,
    td_max_ms: float,
    orientation_tag: str,
    z: Optional[float] = None,
    ext: str = "pdf",
) -> str:
    """Build the figure path for bestfit precession-parameter outputs over (td, I)."""
    orientation_token = str(orientation_tag).strip() or "orientation"
    z_token = _canonical_z_token(z)
    mcz_token = "-".join(_format_min_precision(float(value)) for value in mcz_values)
    stem = "_".join(
        [
            "bestfit_prec_params",
            orientation_token,
            f"z{z_token}",
            f"mcz{mcz_token}",
            f"I{_range_token(I_min, I_max)}",
            f"td{_range_token(td_min_ms, td_max_ms)}",
        ]
    )
    return os.path.join(_ensure_dir(fig_dir), f"{stem}.{ext}")


# ==============================================================================
# (td, I) Inspection And Discovery Helpers
# ==============================================================================


def find_mismatch_I_cube_files(
    results_dir: str,
    td_min_ms: Optional[float],
    td_max_ms: Optional[float],
    mcz_msun: float,
    orientation_tag: str,
    z: Optional[float] = None,
    I_min: Optional[float] = None,
    I_max: Optional[float] = None,
    I_val: Optional[float] = None,
    I_tolerance: float = 1e-6,
    mcz_tolerance: float = 1e-6,
) -> List[str]:
    """Return mismatch I-cube files matching the requested I-td contour run."""
    td_tokens = _td_glob_tokens(td_min_ms, td_max_ms)
    mcz_token = f"mcz{_canonical_token(mcz_msun)}"
    z_token = _canonical_z_token(z)

    patterns = [
        os.path.join(
            results_dir,
            "mismatch_cubes",
            (
                f"mismatch_cubes_z{z_token}_{mcz_token}_I*_{td_token}"
                f"_omega*-*x*_theta*-*x*_gamma0-2pix*_{orientation_tag}.h5"
            ),
        )
        for td_token in td_tokens
    ]
    matches = _glob_union(patterns)
    selected = []
    for path in matches:
        I_parsed = parse_I_from_mismatch_I_cube_path(path)
        if I_parsed is None:
            continue
        if I_val is not None and not _is_close(I_parsed, I_val, I_tolerance):
            continue
        if I_min is not None and I_parsed < I_min - I_tolerance:
            continue
        if I_max is not None and I_parsed > I_max + I_tolerance:
            continue
        selected.append(path)
    return selected


def parse_I_range_from_best_match_I_td_path(path: str) -> Optional[Tuple[float, float]]:
    """Extract (I_min, I_max) from canonical I-td best-match filenames."""
    return _parse_range_token_from_path(path, "_I")


def find_best_match_I_td_file(
    results_dir: str,
    mcz_msun: float,
    I_min: float,
    I_max: float,
    td_min_ms: float,
    td_max_ms: float,
    orientation_tag: str,
    z: Optional[float] = None,
    tolerance: float = 1e-6,
    file_prefix: str = "best_match",
) -> Optional[str]:
    """Return the newest aggregated file for the requested I-td contour run."""
    prefix = str(file_prefix).strip()
    if not prefix:
        raise ValueError("file_prefix must be a non-empty string.")

    z_token = _canonical_z_token(z)
    pattern = os.path.join(
        results_dir,
        "best_match",
        (
            f"{prefix}_z{z_token}_mcz{_canonical_token(mcz_msun)}"
            f"_I{_range_token(I_min, I_max)}x*_td{_range_token(td_min_ms, td_max_ms)}x*"
            f"_omega*-*x*_theta*-*x*_gamma0-2pix*_{orientation_tag}.h5"
        ),
    )
    matches = _glob_union([pattern])
    if not matches:
        return None

    selected = []
    for path in matches:
        parsed = parse_I_range_from_best_match_I_td_path(path)
        if parsed is None:
            continue
        lo, hi = parsed
        if not _is_close(lo, I_min, tolerance):
            continue
        if not _is_close(hi, I_max, tolerance):
            continue

        if not _best_match_file_matches(
            path,
            "I",
            I_min,
            I_max,
            td_min_ms,
            td_max_ms,
            orientation_tag,
            tolerance,
        ):
            continue

        selected.append(path)

    if not selected:
        return None

    selected.sort(key=os.path.getmtime, reverse=True)
    return selected[0]
