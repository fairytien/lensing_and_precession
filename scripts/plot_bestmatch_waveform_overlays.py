import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from modules.plot_utils_v3 import customize_2x1_axes_ratio
from modules.waveform_plotting import plot_best_match_overlay_from_contour


def load_pickle(path: str) -> dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def infer_mcz_label(data: dict, fallback_name: str) -> str:
    mcz_msun = data.get("mcz_msun", None)
    if mcz_msun is not None and np.isfinite(mcz_msun):
        return f"{int(round(float(mcz_msun)))}"
    if "mcz20" in fallback_name:
        return "20"
    if "mcz40" in fallback_name:
        return "40"
    return "unknown"


def plot_one(input_path: str, output_dir: str, f_min: float, npoints: int) -> str:
    data = load_pickle(input_path)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.25)

    summary = plot_best_match_overlay_from_contour(
        data,
        axes,
        f_min=f_min,
        npoints=npoints,
    )
    customize_2x1_axes_ratio(axes)

    title = (
        rf"$\mathcal{{M}}_{{\rm s}}$={summary['mcz_msun']:.3g} $M_\odot$, "
        rf"$\Delta t_d$={summary['td_ms']:.3g} ms, "
        rf"$I$={summary['I']:.3g}, "
        rf"$\tilde{{\Omega}}$={summary['omega_tilde']:.3g}, "
        rf"$\tilde{{\theta}}$={summary['theta_tilde']:.3g}, "
        rf"$\gamma_P$={summary['gamma_P']:.3g}, "
        rf"$\epsilon_\min$={summary['epsilon']:.3g}"
    )
    fig.suptitle(title, fontsize=22, y=1.03)

    filename_root = os.path.basename(input_path)
    mcz_label = infer_mcz_label(data, filename_root)
    out_path = os.path.join(output_dir, f"sys2_waveforms_mcz{mcz_label}_fracamp.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(
        "Best match:",
        f"omega_tilde={summary['omega_tilde']:.6g},",
        f"theta_tilde={summary['theta_tilde']:.6g},",
        f"gamma_P={summary['gamma_P']:.6g},",
        f"epsilon={summary['epsilon']:.6g}",
    )

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Overlay best-match RP waveform on lensed waveform for contour pickles, "
            "with fractional amplitude change and phase difference panels."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=[
            "data/indiv_contours/v3_indiv_contour_mcz20_td30ms_I0.5_thetaS0.785_phiS0.0_thetaJ1.571_phiJ1.571_z0.2_2026-03-03_15-13-55.pkl",
            "data/indiv_contours/v3_indiv_contour_mcz40_td30ms_I0.5_thetaS0.785_phiS0.0_thetaJ1.571_phiJ1.571_z0.2_2026-03-03_15-10-52.pkl",
        ],
        help="One or more contour pickle paths.",
    )
    parser.add_argument(
        "--output_dir",
        default="figures/waveforms",
        help="Directory for output PDF files.",
    )
    parser.add_argument("--f_min", type=float, default=20.0)
    parser.add_argument("--npoints", type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for input_path in args.input:
        plot_one(input_path, args.output_dir, args.f_min, args.npoints)


if __name__ == "__main__":
    main()
