import os
import re
import argparse
import pickle
import sys
from typing import Optional, Tuple

import numpy as np
import h5py

# Use non-interactive backend for cluster environments
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Ensure project root is importable when script is launched directly.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.plot_utils import (
    apply_physics_paper_style,
    format_colorbar_ticks,
    save_figure,
)
from modules.default_params import SOLMASS2SEC

apply_physics_paper_style()


X_LABEL_OMEGA = r"$\tilde{\Omega}$"
Y_LABEL_THETA = r"$\tilde{\theta}$"
X_LABEL_TD = r"$\Delta t_{\mathrm{d}}\,[\mathrm{ms}]$"
Y_LABEL_MCZ = r"$\mathcal{M}_{\mathrm{s}}\,[\mathrm{M}_\odot]$"
COLORBAR_LABEL = (
    r"$\epsilon(\tilde{\mathit{h}}_{\mathrm{L}}, \tilde{\mathit{h}}_{\mathrm{P}})$"
)


def derive_fig_path_from_input(input_path: str, tag: Optional[str] = None) -> str:
    figures_dir = os.path.join(PROJECT_ROOT, "figures", "contour_omega_theta")
    os.makedirs(figures_dir, exist_ok=True)

    filename = os.path.basename(input_path)
    # Strip the trailing timestamp _YYYY-MM-DD_HH-MM-SS if present
    name_wo_ext = os.path.splitext(filename)[0]
    m = re.match(r"^(.*)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", name_wo_ext)
    base = m.group(1) if m else name_wo_ext
    if tag:
        base = f"{base}_{tag}"
    return os.path.join(figures_dir, f"{base}.pdf")


def _as_2d_float_array(name: str, arr: np.ndarray) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{name} must be a numpy array")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    return np.asarray(arr, dtype=np.float64)


def _infer_chirp_mass_msun_from_pickle(out: dict) -> Optional[float]:
    mcz_msun = out.get("mcz_msun")
    if mcz_msun is not None and np.isfinite(mcz_msun):
        return float(mcz_msun)

    source_params = out.get("source_params")
    if isinstance(source_params, dict):
        mcz_sec = source_params.get("mcz")
        if mcz_sec is not None and np.isfinite(mcz_sec):
            return float(mcz_sec) / SOLMASS2SEC
    return None


def _load_contour_from_pickle(path: str):
    with open(path, "rb") as f:
        out = pickle.load(f)

    required = {"omega_matrix", "theta_matrix", "epsilon_matrix"}
    missing = [k for k in required if k not in out]
    if missing:
        raise KeyError(f"Missing required keys in pickle: {missing}")

    X = _as_2d_float_array("omega_matrix", out["omega_matrix"])
    Y = _as_2d_float_array("theta_matrix", out["theta_matrix"])
    Z = _as_2d_float_array("epsilon_matrix", out["epsilon_matrix"])
    if X.shape != Y.shape or X.shape != Z.shape:
        raise ValueError(f"Mismatched shapes: X{X.shape}, Y{Y.shape}, Z{Z.shape}")

    return {
        "X": X,
        "Y": Y,
        "Z": Z,
        "x_label": X_LABEL_OMEGA,
        "y_label": Y_LABEL_THETA,
        "chirp_mass": _infer_chirp_mass_msun_from_pickle(out),
        "chirp_mass_range": None,
    }


def _load_contour_from_h5(path: str, td_ms: Optional[float] = None):
    with h5py.File(path, "r") as h5:
        keys = set(h5.keys())

        if {"omega_matrix", "theta_matrix", "epsilon_matrix"}.issubset(keys):
            X = _as_2d_float_array("omega_matrix", np.array(h5["omega_matrix"]))
            Y = _as_2d_float_array("theta_matrix", np.array(h5["theta_matrix"]))
            Z = _as_2d_float_array("epsilon_matrix", np.array(h5["epsilon_matrix"]))
            if X.shape != Y.shape or X.shape != Z.shape:
                raise ValueError(
                    f"Mismatched shapes: X{X.shape}, Y{Y.shape}, Z{Z.shape}"
                )

            mcz_value = None
            if "source_param_mcz" in h5.attrs:
                arr = np.asarray(h5.attrs["source_param_mcz"])
                if arr.size == 1 and np.isfinite(arr.item()):
                    mcz_value = arr.item() / SOLMASS2SEC

            return {
                "X": X,
                "Y": Y,
                "Z": Z,
                "x_label": X_LABEL_OMEGA,
                "y_label": Y_LABEL_THETA,
                "chirp_mass": mcz_value,
                "chirp_mass_range": None,
            }

        if {"mcz", "td", "epsilon_min"}.issubset(keys):
            mcz = np.asarray(h5["mcz"], dtype=np.float64)
            td_ms_arr = np.asarray(h5["td"], dtype=np.float64) * 1e3
            Z = _as_2d_float_array("epsilon_min", np.array(h5["epsilon_min"]))

            X, Y = np.meshgrid(td_ms_arr, mcz)
            if X.shape != Z.shape or Y.shape != Z.shape:
                raise ValueError(
                    f"Mismatched shapes after meshgrid: X{X.shape}, Y{Y.shape}, Z{Z.shape}"
                )

            mcz_range = (float(np.nanmin(mcz)), float(np.nanmax(mcz)))
            return {
                "X": X,
                "Y": Y,
                "Z": Z,
                "x_label": X_LABEL_TD,
                "y_label": Y_LABEL_MCZ,
                "chirp_mass": None,
                "chirp_mass_range": mcz_range,
            }

        if {"td", "theta", "omega", "epsilon_min_grid"}.issubset(keys):
            if td_ms is None:
                raise ValueError(
                    "Cube schema (td, theta, omega, epsilon_min_grid) requires --td_ms"
                )
            td = np.asarray(h5["td"], dtype=np.float64)
            theta = np.asarray(h5["theta"], dtype=np.float64)
            omega = np.asarray(h5["omega"], dtype=np.float64)
            eps_grid = np.asarray(h5["epsilon_min_grid"], dtype=np.float64)
            j = int(np.argmin(np.abs(td * 1e3 - td_ms)))
            Z = eps_grid[j]  # (n_theta, n_omega)
            O, T = np.meshgrid(omega, theta)
            mcz_val = None
            if "mcz" in keys:
                arr = np.asarray(h5["mcz"], dtype=np.float64).ravel()
                if arr.size == 1 and np.isfinite(arr[0]):
                    mcz_val = float(arr[0])
            return {
                "X": O,
                "Y": T,
                "Z": Z,
                "x_label": X_LABEL_OMEGA,
                "y_label": Y_LABEL_THETA,
                "chirp_mass": mcz_val,
                "chirp_mass_range": None,
            }

        raise ValueError(
            "Unsupported HDF5 schema. Expected one of: "
            "(omega_matrix, theta_matrix, epsilon_matrix), "
            "(mcz, td, epsilon_min), or "
            "(td, theta, omega, epsilon_min_grid) [cube, requires --td_ms]. "
            f"Found keys: {sorted(keys)}"
        )


def load_contour_dataset(path: str, td_ms: Optional[float] = None):
    ext = os.path.splitext(path)[1].lower()
    if ext in {".pkl", ".pickle"}:
        return _load_contour_from_pickle(path)
    if ext in {".h5", ".hdf5"}:
        return _load_contour_from_h5(path, td_ms=td_ms)
    raise ValueError(
        f"Unsupported input extension '{ext}'. Use .pkl/.pickle or .h5/.hdf5"
    )


def _add_chirp_mass_box(
    ax, mcz_value: Optional[float], mcz_range: Optional[Tuple[float, float]]
) -> None:
    if mcz_value is not None and np.isfinite(mcz_value):
        label = (
            rf"$\mathcal{{M}}_{{\mathrm{{s}}}} = {mcz_value:.3g}\,\mathrm{{M}}_\odot$"
        )
    elif mcz_range is not None:
        m_min, m_max = mcz_range
        if not (np.isfinite(m_min) and np.isfinite(m_max)):
            return
        if np.isclose(m_min, m_max, rtol=0, atol=1e-12):
            label = (
                rf"$\mathcal{{M}}_{{\mathrm{{s}}}} = {m_min:.3g}\,\mathrm{{M}}_\odot$"
            )
        else:
            label = (
                rf"$\mathcal{{M}}_{{\mathrm{{s}}}} \in [{m_min:.3g}, {m_max:.3g}]"
                rf"\,\mathrm{{M}}_\odot$"
            )
    else:
        return

    dummy = Line2D([], [], linestyle="None")
    legend = ax.legend(
        [dummy],
        [label],
        loc="upper left",
        fontsize=11,
        frameon=True,
        framealpha=0.55,
        facecolor="white",
        edgecolor="black",
        fancybox=True,
        handlelength=0.0,
        handletextpad=0.0,
        borderpad=0.35,
    )
    for handle in getattr(legend, "legendHandles", []):
        handle.set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render contour plot from saved contour output (.pkl or .h5), "
            "inferring axes from dataset schema."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to contour input (.pkl/.pickle or .h5/.hdf5).",
    )
    parser.add_argument(
        "--fig",
        default=None,
        help="Optional output figure path (.pdf/.png). If omitted, derived from input name.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional tag appended to derived figure basename (default: empty).",
    )
    parser.add_argument(
        "--theta_points",
        type=int,
        default=None,
        help=(
            "If set and smaller than the stored grid, subsample the theta dimension (rows) "
            "to this many points before plotting."
        ),
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=100,
        help="Number of contour levels for plotting (default: 100).",
    )
    parser.add_argument(
        "--cbar-decimals",
        "--cbar-sigfigs",
        type=int,
        default=2,
        help="Decimal places for colorbar tick labels (default: 2).",
    )
    parser.add_argument(
        "--td_ms",
        type=float,
        default=None,
        help="Time-delay slice in ms; required when input is a mismatch cube (td, theta, omega, epsilon_min_grid).",
    )
    parser.add_argument(
        "--cbar-n-ticks",
        type=int,
        default=10,
        help="Target number of colorbar ticks (default: 10).",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    loaded = load_contour_dataset(input_path, td_ms=args.td_ms)
    X = loaded["X"]
    Y = loaded["Y"]
    Z = loaded["Z"]
    x_label = loaded["x_label"]
    y_label = loaded["y_label"]
    chirp_mass = loaded["chirp_mass"]
    chirp_mass_range = loaded["chirp_mass_range"]

    # Optional theta subsampling (reduce number of rows)
    tag = args.tag
    if (
        args.theta_points is not None
        and isinstance(X, np.ndarray)
        and args.theta_points > 0
        and args.theta_points < X.shape[0]
    ):
        n_theta_orig = X.shape[0]
        idx = np.linspace(0, n_theta_orig - 1, args.theta_points, dtype=int)
        idx = np.unique(idx)
        if idx.size < args.theta_points:
            # In rare cases of duplicates due to rounding, pad by adding missing indices
            missing = args.theta_points - idx.size
            extra = np.setdiff1d(np.arange(n_theta_orig), idx)[:missing]
            idx = np.sort(np.concatenate([idx, extra]))
        print(
            f"Subsampling theta dimension from {n_theta_orig} to {idx.size} rows for plotting."
        )
        X = X[idx, :]
        Y = Y[idx, :]
        Z = Z[idx, :]
        if not tag:
            tag = f"theta{idx.size}"

    # Add levels tag if not default
    if args.levels != 100:
        levels_tag = f"levels{args.levels}"
        tag = f"{tag}_{levels_tag}" if tag else levels_tag

    fig_path = (
        args.fig
        if args.fig
        else derive_fig_path_from_input(input_path, tag=tag if tag else None)
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    cf = ax.contourf(X, Y, Z, levels=args.levels, cmap="jet")

    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label(COLORBAR_LABEL)
    vmin = float(np.nanmin(Z))
    vmax = float(np.nanmax(Z))
    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        format_colorbar_ticks(
            cbar,
            vmin,
            vmax,
            n_ticks=max(2, int(args.cbar_n_ticks)),
            decimals=max(0, args.cbar_decimals),
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _add_chirp_mass_box(ax, chirp_mass, chirp_mass_range)

    save_figure(fig, fig_path)


if __name__ == "__main__":
    main()
