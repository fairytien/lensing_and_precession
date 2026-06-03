from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional, Tuple, cast

import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from modules.bank_io import read_best_match_mcz_td_data
from modules.filenames import _canonical_token, _range_token
from modules.lens_cycle_extrema import find_mcz_peaks, find_mcz_troughs
from modules.plot_utils import (
    apply_physics_paper_style,
    save_figure,
    LBL_EPS_LNP,
    LBL_MIN_MCZ_EPS_LNP,
    LBL_EPS_LRP,
)
from modules.waveform import mcz_for_n_lens_cycles
from scripts.utils.plot_cycles_and_extrema import FIXED_MCZ_CYCLE_STYLES

# Extrema overlay style conventions mirror scripts/utils/plot_cycles_and_extrema.py.
PEAK_COLOR = "magenta"
TROUGH_COLOR = "cyan"
EXTREMA_LINESTYLE = ":"


def _normalize_orientation_tag(tag: str) -> str:
    normalized = str(tag).strip().replace(".", "_")
    if normalized.startswith("Taman_"):
        return normalized
    if normalized.startswith("Taman") and "_" not in normalized:
        # Handle forms like "Tamanedgeon" defensively.
        suffix = normalized[len("Taman") :].lstrip("._")
        if suffix:
            return f"Taman_{suffix}"
    return normalized


def _nearest_index(arr: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(arr - target)))


def _extract_rp_curve_from_best_match(
    best_match_path: str,
    td_target_s: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[float], str]:
    ds = read_best_match_mcz_td_data(best_match_path, "epsilon_min")
    mcz_arr = np.asarray(ds["mcz"], dtype=float)
    td_arr = np.asarray(ds["td"], dtype=float)
    eps = np.asarray(ds["values"], dtype=float)
    z_val = cast(Optional[float], ds.get("z"))
    orientation_tag = _normalize_orientation_tag(str(ds["orientation_tag"]))

    td_idx = _nearest_index(td_arr, td_target_s)
    curve = np.asarray(eps[:, td_idx], dtype=float)
    return mcz_arr, curve, z_val, orientation_tag


def _default_output_path(
    I_val: float,
    td_ms: float,
    z_val: float,
    mcz_min: float,
    mcz_max: float,
    orientation_tag: str,
) -> str:

    I_str = f"I{_canonical_token(I_val)}"
    td_str = f"td{int(td_ms)}"
    z_str = f"z{_canonical_token(z_val)}"
    mcz_str = f"mcz{_range_token(mcz_min, mcz_max)}"
    filename = (
        f"compare_LvsNP_RP_{I_str}_{td_str}_{z_str}_{mcz_str}_{orientation_tag}.pdf"
    )
    return os.path.join("figures/contour_mcz_td", filename)


def _build_curves(
    l_np_path: str,
    l_np_opt_path: str,
    td_ms: float,
    rp_best_match: str,
    z: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
]:
    l_np = read_best_match_mcz_td_data(l_np_path, "epsilon_min")
    mcz_fixed = np.asarray(l_np["mcz"], dtype=float)
    td_arr = np.asarray(l_np["td"], dtype=float)

    td_target_s = td_ms / 1e3
    td_idx = _nearest_index(td_arr, td_target_s)
    td_actual_s = float(td_arr[td_idx])

    fixed_np_curve = np.asarray(l_np["values"][:, td_idx], dtype=float)

    z_lnp = cast(Optional[float], l_np["z"])
    if z_lnp is None or not np.isclose(z_lnp, z, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"L-vs-NP contour redshift mismatch: file z={z_lnp}, requested z={z}. "
            "Provide an L-vs-NP contour file generated at the requested redshift."
        )

    l_np_opt = read_best_match_mcz_td_data(l_np_opt_path, "epsilon_min")
    mcz_opt = np.asarray(l_np_opt["mcz"], dtype=float)
    td_arr_opt = np.asarray(l_np_opt["td"], dtype=float)
    td_idx_opt = _nearest_index(td_arr_opt, td_target_s)
    opt_np_curve = np.asarray(l_np_opt["values"][:, td_idx_opt], dtype=float)

    z_lnp_opt = cast(Optional[float], l_np_opt["z"])
    if z_lnp_opt is None or not np.isclose(z_lnp_opt, z, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"L-vs-NP optimized contour redshift mismatch: file z={z_lnp_opt}, requested z={z}."
        )

    if not os.path.isfile(rp_best_match):
        raise FileNotFoundError(f"RP best-match file not found: {rp_best_match}")
    mcz_rp, blue_curve, z_rp, orientation_tag = _extract_rp_curve_from_best_match(
        best_match_path=rp_best_match,
        td_target_s=td_actual_s,
    )
    if z_rp is None or not np.isclose(z_rp, z, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"RP best_match redshift mismatch: file z={z_rp}, requested z={z}. "
            "Provide a best_match file generated at the requested redshift."
        )

    with h5py.File(l_np_path, "r") as h5:
        eta_val = float(h5.attrs.get("source_param_eta", 0.25))

    z_actual = float(z_lnp)

    meta = {
        "I": float(l_np["I"]),
        "z": z_actual,
        "td_s": td_actual_s,
        "eta": eta_val,
        "orientation_tag": orientation_tag,
        "rp_source": f"best_match:{rp_best_match}",
    }
    return mcz_fixed, mcz_opt, mcz_rp, fixed_np_curve, opt_np_curve, blue_curve, meta


def _plot(
    mcz_fixed: np.ndarray,
    mcz_opt: np.ndarray,
    mcz_rp: np.ndarray,
    fixed_np_curve: np.ndarray,
    opt_np_curve: np.ndarray,
    blue_curve: np.ndarray,
    meta: Dict[str, Any],
    output_path: str,
    z: float = 0.0,
    one_col_legend: bool = False,
) -> None:
    apply_physics_paper_style(base_font=16, label_font=20, tick_font=17, legend_font=14)
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    ax.set_facecolor("white")

    ax.plot(
        mcz_fixed,
        fixed_np_curve,
        color="red",
        lw=2.2,
        ls="-",
        label=LBL_EPS_LNP,
        zorder=4,
    )
    ax.plot(
        mcz_opt,
        opt_np_curve,
        color="green",
        lw=2.2,
        ls="--",
        label=LBL_MIN_MCZ_EPS_LNP,
        zorder=4,
    )
    ax.plot(
        mcz_rp,
        blue_curve,
        color="blue",
        lw=2.2,
        ls="-",
        label=LBL_EPS_LRP,
        zorder=4,
    )

    td_s = float(meta["td_s"])
    eta = float(meta["eta"])
    z_scale = 1.0 + float(meta["z"])
    mcz_min_src = float(mcz_fixed.min())
    mcz_max_src = float(mcz_fixed.max())
    # Detector-frame bounds for cycle/extrema helpers (they operate in detector frame).
    mcz_min_det = mcz_min_src * z_scale
    mcz_max_det = mcz_max_src * z_scale

    # Cycle-line convention: black with -, --, : for N_lensed=1,2,3.
    for n_cycles, ls_style in FIXED_MCZ_CYCLE_STYLES.items():
        x_val_src = (
            float(mcz_for_n_lens_cycles(float(n_cycles), td_s, f_min=20.0, eta=eta))
            / z_scale
        )
        if mcz_min_src <= x_val_src <= mcz_max_src:
            ax.axvline(
                x=x_val_src, color="black", ls=ls_style, lw=1.6, alpha=0.9, zorder=2
            )

    # Extrema-line convention: peaks=magenta dotted, troughs=cyan dotted.
    for x_val in (
        find_mcz_peaks(
            np.array([td_s]), eta=eta, mcz_min=mcz_min_det, mcz_max=mcz_max_det
        )[1]
        / z_scale
    ):
        ax.axvline(
            x=float(x_val),
            color=PEAK_COLOR,
            ls=EXTREMA_LINESTYLE,
            lw=1.6,
            alpha=0.95,
            zorder=2,
        )

    for x_val in (
        find_mcz_troughs(
            np.array([td_s]), eta=eta, mcz_min=mcz_min_det, mcz_max=mcz_max_det
        )[1]
        / z_scale
    ):
        ax.axvline(
            x=float(x_val),
            color=TROUGH_COLOR,
            ls=EXTREMA_LINESTYLE,
            lw=1.6,
            alpha=0.95,
            zorder=2,
        )

    ep_horizontal = 1.0 - (1.0 + float(meta["I"])) ** (-0.5)
    ax.axhline(
        y=ep_horizontal,
        color="gray",
        ls="-.",
        lw=2.0,
        label=r"$1 - (1 + I)^{-1/2}$",
        zorder=2,
    )

    ax.set_xlim(mcz_min_src - 1.0, mcz_max_src + 1.0)
    y_max = float(np.nanmax(np.concatenate([fixed_np_curve, opt_np_curve, blue_curve])))
    ax.set_ylim(-0.01, max(0.24, y_max + 0.01))

    ax.set_xlabel(r"$\mathcal{M}_{\mathrm{s}}\,[M_\odot]$")
    ax.set_ylabel(r"$\epsilon$")



    cycle_legend_handles = []
    for n_cycles, ls_style in FIXED_MCZ_CYCLE_STYLES.items():
        cycle_legend_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                ls=ls_style,
                lw=1.6,
                label=rf"$N_{{\mathrm{{lensed}}}}={n_cycles}$",
            )
        )

    extrema_legend_handles = [
        Line2D(
            [0],
            [0],
            color=PEAK_COLOR,
            ls=EXTREMA_LINESTYLE,
            lw=1.6,
            label=r"$\mathrm{peak}$",
        ),
        Line2D(
            [0],
            [0],
            color=TROUGH_COLOR,
            ls=EXTREMA_LINESTYLE,
            lw=1.6,
            label=r"$\mathrm{trough}$",
        ),
    ]

    handles, labels = ax.get_legend_handles_labels()
    if one_col_legend:
        all_handles = handles + cycle_legend_handles + extrema_legend_handles
        ax.legend(handles=all_handles, loc="best", framealpha=0.85)
    else:
        col1 = handles
        col2 = cycle_legend_handles + extrema_legend_handles

        n_rows = max(len(col1), len(col2))
        dummy = Line2D([], [], color="none", label="")

        padded_col1 = list(col1) + [dummy] * (n_rows - len(col1))
        padded_col2 = list(col2) + [dummy] * (n_rows - len(col2))

        all_handles = padded_col1 + padded_col2
        ax.legend(handles=all_handles, ncol=2, loc="best", framealpha=0.85)

    save_figure(fig, output_path, dpi=300)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a fixed-td mcz slice comparing L-vs-NP and L-vs-RP mismatch curves "
            "using modern pipeline artifacts (L-vs-RP from best_match HDF5)."
        )
    )
    parser.add_argument(
        "--l-np-contour",
        type=str,
        default=(
            "data/contour_mcz_td/"
            "contour_L_NP_I0.5_z1_mcz10-90Msun_td20-70ms_min_mismatch_Taman_edgeon.h5"
        ),
        help="Path to L-vs-NP contour HDF5 (fixed mcz).",
    )
    parser.add_argument(
        "--l-np-opt-contour",
        type=str,
        required=True,
        help="Path to L-vs-NP contour HDF5 (optimized over template mcz).",
    )
    parser.add_argument(
        "--rp-best-match",
        type=str,
        required=True,
        help="RP best-match HDF5 path.",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=1.0,
        help="Common redshift used by all plotted data sources (default: 1).",
    )
    parser.add_argument(
        "--td-ms",
        type=float,
        default=30.0,
        help="Time-delay slice in ms (nearest available grid bin is used).",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--one-col-legend",
        action="store_true",
        help="Use a single-column layout for the legend instead of two columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mcz_fixed, mcz_opt, mcz_rp, fixed_np_curve, opt_np_curve, blue_curve, meta = _build_curves(
        l_np_path=args.l_np_contour,
        l_np_opt_path=args.l_np_opt_contour,
        td_ms=args.td_ms,
        rp_best_match=args.rp_best_match,
        z=args.z,
    )

    output_path = args.output
    if output_path is None:
        output_path = _default_output_path(
            I_val=float(meta["I"]),
            td_ms=args.td_ms,
            z_val=float(meta["z"]),
            mcz_min=mcz_fixed.min(),
            mcz_max=mcz_fixed.max(),
            orientation_tag=str(meta["orientation_tag"]),
        )

    _plot(
        mcz_fixed=mcz_fixed,
        mcz_opt=mcz_opt,
        mcz_rp=mcz_rp,
        fixed_np_curve=fixed_np_curve,
        opt_np_curve=opt_np_curve,
        blue_curve=blue_curve,
        meta=meta,
        output_path=output_path,
        z=args.z,
        one_col_legend=args.one_col_legend,
    )

    print(f"RP source used: {meta['rp_source']}")


if __name__ == "__main__":
    main()
