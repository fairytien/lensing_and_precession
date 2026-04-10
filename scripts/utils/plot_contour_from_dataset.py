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
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# Ensure project root is importable when script is launched directly.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.plot_utils_v3 import apply_physics_paper_style
from modules.default_params_v3 import SOLMASS2SEC

apply_physics_paper_style()


X_LABEL_OMEGA = r"$\tilde{\Omega}$"
Y_LABEL_THETA = r"$\tilde{\theta}$"
X_LABEL_TD = r"$\Delta t_{\mathrm{d}}\,[\mathrm{ms}]$"
Y_LABEL_MCZ = r"$\mathcal{M}_{\mathrm{s}}\,[\mathrm{M}_\odot]$"
COLORBAR_LABEL = (
    r"$\epsilon(\tilde{\mathit{h}}_{\mathrm{L}}, \tilde{\mathit{h}}_{\mathrm{RP}})$"
)


def _find_data_root(path: str) -> Optional[str]:
    cur = os.path.abspath(path)
    while True:
        if os.path.basename(cur) == "data":
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def derive_fig_path_from_input(input_path: str, tag: Optional[str] = None) -> str:
    data_dir = os.path.dirname(input_path)
    data_root = _find_data_root(data_dir)
    if data_root is not None:
        figures_dir = os.path.join(data_root, "figures", "utils")
    else:
        root_dir = os.path.dirname(data_dir)
        figures_dir = os.path.join(root_dir, "figures", "utils")
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


def _optional_scalar_float(value) -> Optional[float]:
    arr = np.asarray(value)
    if arr.size != 1:
        return None
    scalar = float(arr.reshape(()))
    if not np.isfinite(scalar):
        return None
    return scalar


def _chirp_mass_box_text(
    *, mcz_value: Optional[float], mcz_range: Optional[Tuple[float, float]]
) -> Optional[str]:
    if mcz_value is not None and np.isfinite(mcz_value):
        return (
            rf"$\mathcal{{M}}_{{\mathrm{{s}}}} = {mcz_value:.3g}\,\mathrm{{M}}_\odot$"
        )

    if mcz_range is not None:
        m_min, m_max = mcz_range
        if np.isfinite(m_min) and np.isfinite(m_max):
            if np.isclose(m_min, m_max, rtol=0, atol=1e-12):
                return rf"$\mathcal{{M}}_{{\mathrm{{s}}}} = {m_min:.3g}\,\mathrm{{M}}_\odot$"
            return (
                rf"$\mathcal{{M}}_{{\mathrm{{s}}}} \in [{m_min:.3g}, {m_max:.3g}]"
                rf"\,\mathrm{{M}}_\odot$"
            )
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


def _load_contour_from_h5(path: str):
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
                attr_value = _optional_scalar_float(h5.attrs["source_param_mcz"])
                if attr_value is not None:
                    mcz_value = attr_value / SOLMASS2SEC

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
            td_ms = np.asarray(h5["td"], dtype=np.float64) * 1e3
            Z = _as_2d_float_array("epsilon_min", np.array(h5["epsilon_min"]))

            X, Y = np.meshgrid(td_ms, mcz)
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

        raise ValueError(
            "Unsupported HDF5 schema. Expected either "
            "(omega_matrix, theta_matrix, epsilon_matrix) or (mcz, td, epsilon_min). "
            f"Found keys: {sorted(keys)}"
        )


def load_contour_dataset(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in {".pkl", ".pickle"}:
        return _load_contour_from_pickle(path)
    if ext in {".h5", ".hdf5"}:
        return _load_contour_from_h5(path)
    raise ValueError(
        f"Unsupported input extension '{ext}'. Use .pkl/.pickle or .h5/.hdf5"
    )


def _add_chirp_mass_box(
    ax, mcz_value: Optional[float], mcz_range: Optional[Tuple[float, float]]
) -> None:
    label = _chirp_mass_box_text(mcz_value=mcz_value, mcz_range=mcz_range)
    if not label:
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input",
        default=None,
        help="Path to contour input (.pkl/.pickle or .h5/.hdf5).",
    )
    group.add_argument(
        "--pkl",
        default=None,
        help="Backward-compatible alias for --input when using pickle files.",
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
        "--dpi",
        type=int,
        default=200,
        help="Output resolution (default: 200).",
    )
    parser.add_argument(
        "--cbar-n-ticks",
        type=int,
        default=10,
        help="Target number of colorbar ticks (default: 10).",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input if args.input else args.pkl)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    loaded = load_contour_dataset(input_path)
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
        tick_count = max(2, int(args.cbar_n_ticks))
        cbar.set_ticks(np.linspace(vmin, vmax, tick_count))
    else:
        cbar.locator = mticker.MaxNLocator(nbins=10, steps=[1, 2, 2.5, 5, 10])
    cbar.formatter = mticker.FormatStrFormatter(f"%.{max(0, args.cbar_decimals)}f")
    cbar.update_ticks()

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _add_chirp_mass_box(ax, chirp_mass, chirp_mass_range)

    fig.tight_layout(pad=0.2)
    fig.savefig(fig_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("Figure saved as", fig_path)


if __name__ == "__main__":
    main()
