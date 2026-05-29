from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np

from modules.bank_io import read_best_match_mcz_td_contour_data
from modules.default_params import lens_params_1
from modules.filenames import compare_mcz_td_figure_filename
from modules.lens_cycle_extrema import find_mcz_peaks, find_mcz_troughs
from modules.plot_utils import apply_physics_paper_style, save_figure
from modules.waveform import mcz_for_n_lens_cycles
from scripts.utils.plot_cycles_and_extrema import FIXED_MCZ_CYCLE_STYLES

# Extrema overlay style conventions mirror scripts/utils/plot_cycles_and_extrema.py.
PEAK_COLOR = "magenta"
TROUGH_COLOR = "white"
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
    ds = read_best_match_mcz_td_contour_data(best_match_path, "epsilon_min")
    mcz_arr = np.asarray(ds["mcz"], dtype=float)
    td_arr = np.asarray(ds["td"], dtype=float)
    eps = np.asarray(ds["values"], dtype=float)
    z_val = cast(Optional[float], ds.get("z"))
    orientation_tag = _normalize_orientation_tag(str(ds["orientation_tag"]))

    td_idx = _nearest_index(td_arr, td_target_s)
    curve = np.asarray(eps[:, td_idx], dtype=float)
    return mcz_arr, curve, z_val, orientation_tag


def _default_output_path(
    I_val: float, z_val: Optional[float], orientation_tag: str
) -> str:
    base = compare_mcz_td_figure_filename(
        fig_dir="figures/contour_mcz_td",
        I=I_val,
        z=z_val,
        orientation_tags=[orientation_tag],
        ext="pdf",
    )
    stem, ext = os.path.splitext(base)
    return f"{stem}_mcz_slice_l_np_rp.{ext.lstrip('.')}"


def _build_curves(
    l_np_path: str,
    td_ms: float,
    rp_best_match: str,
    z: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    l_np = read_best_match_mcz_td_contour_data(l_np_path, "epsilon_min")

    mcz_arr = np.asarray(l_np["mcz"], dtype=float)
    td_arr = np.asarray(l_np["td"], dtype=float)

    td_target_s = td_ms / 1e3
    td_idx = _nearest_index(td_arr, td_target_s)
    td_actual_s = float(td_arr[td_idx])

    red_curve = np.asarray(l_np["values"][:, td_idx], dtype=float)

    z_lnp = cast(Optional[float], l_np["z"])
    if z_lnp is None or not np.isclose(z_lnp, z, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"L-vs-NP contour redshift mismatch: file z={z_lnp}, requested z={z}. "
            "Provide an L-vs-NP contour file generated at the requested redshift."
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

    meta = {
        "I": float(l_np["I"]),
        "z": z,
        "td_s": td_actual_s,
        "eta": float(lens_params_1["eta"]),
        "orientation_tag": orientation_tag,
        "rp_source": f"best_match:{rp_best_match}",
    }
    return mcz_arr, mcz_rp, red_curve, blue_curve, meta


def _plot(
    mcz_arr: np.ndarray,
    mcz_rp: np.ndarray,
    red_curve: np.ndarray,
    blue_curve: np.ndarray,
    meta: Dict[str, Any],
    output_path: str,
    z: float = 0.0,
) -> None:
    apply_physics_paper_style(base_font=16, label_font=20, tick_font=17, legend_font=16)
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    ax.set_facecolor("#d9d9d9")

    ax.plot(
        mcz_arr,
        red_curve,
        color="red",
        lw=2.2,
        ls="--",
        label=r"$\min_{\mathcal{M}_{\mathrm{t}}}\,\epsilon(\tilde{h}_{\mathrm{L}}, \tilde{h}_{\mathrm{NP}})$",
    )
    ax.plot(
        mcz_rp,
        blue_curve,
        color="blue",
        lw=2.2,
        label=r"$\epsilon(\tilde{h}_{\mathrm{L}}, \tilde{h}_{\mathrm{RP}})$",
    )

    td_s = float(meta["td_s"])
    eta = float(meta["eta"])
    z_scale = 1.0 + z
    all_mcz = np.concatenate([mcz_arr, mcz_rp])
    mcz_min_src = float(np.nanmin(all_mcz))
    mcz_max_src = float(np.nanmax(all_mcz))
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

    # Extrema-line convention: peaks=magenta dotted, troughs=white dotted.
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
    ax.axhline(y=ep_horizontal, color="gray", ls=":", lw=2.0)

    ax.set_xlim(mcz_min_src - 1.0, mcz_max_src + 1.0)
    y_max = float(np.nanmax(np.concatenate([red_curve, blue_curve])))
    ax.set_ylim(-0.01, max(0.24, y_max + 0.01))

    ax.set_xlabel(r"$\mathcal{M}_{\mathrm{s}}\,[M_\odot]$")
    ax.set_ylabel(r"$\epsilon$")
    ax.legend(loc="center right", framealpha=0.85)

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mcz_arr, mcz_rp, red_curve, blue_curve, meta = _build_curves(
        l_np_path=args.l_np_contour,
        td_ms=args.td_ms,
        rp_best_match=args.rp_best_match,
        z=args.z,
    )

    output_path = args.output
    if output_path is None:
        output_path = _default_output_path(
            I_val=float(meta["I"]),
            z_val=args.z,
            orientation_tag=str(meta["orientation_tag"]),
        )

    _plot(
        mcz_arr=mcz_arr,
        mcz_rp=mcz_rp,
        red_curve=red_curve,
        blue_curve=blue_curve,
        meta=meta,
        output_path=output_path,
        z=args.z,
    )

    print(f"RP source used: {meta['rp_source']}")


if __name__ == "__main__":
    main()
