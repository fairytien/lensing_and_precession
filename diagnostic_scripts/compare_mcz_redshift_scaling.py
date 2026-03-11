#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from matplotlib.lines import Line2D


DEFAULT_Z0_PATH = (
    "data/super_contours/"
    "contour_L_NP_I0.5_z1e-06_mcz10-90Msun_td20-70ms_min_mismatch_Taman.edgeon.h5"
)
DEFAULT_Z1_PATH = (
    "data/super_contours/"
    "contour_L_NP_I0.5_z1_mcz5-45Msun_td20-70ms_min_mismatch_Taman.edgeon.h5"
)
DEFAULT_OUTDIR = "figures/super_contours"
DEFAULT_BASENAME = "mcz_redshift_scaling_edgeon"
DEFAULT_LEVELS = [1e-3, 3e-3, 1e-2, 3e-2]


def load_contour(path: str) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {
            "mcz": handle["mcz_arr"][:],
            "td": handle["td_arr"][:],
            "epsilon": handle["epsilon_matrix"][:],
        }


def compute_comparison(
    z0_data: dict[str, np.ndarray], z1_data: dict[str, np.ndarray], z0: float
) -> dict:
    scale = (1.0 + z0) / 2.0
    scaled_mcz = z0_data["mcz"] * scale
    interpolator = RegularGridInterpolator(
        (scaled_mcz, z0_data["td"]),
        z0_data["epsilon"],
        bounds_error=False,
        fill_value=np.nan,
    )

    mcz_grid, td_grid = np.meshgrid(z1_data["mcz"], z1_data["td"], indexing="ij")
    z0_on_z1 = interpolator(
        np.column_stack([mcz_grid.ravel(), td_grid.ravel()])
    ).reshape(z1_data["epsilon"].shape)

    mask = np.isfinite(z0_on_z1) & np.isfinite(z1_data["epsilon"])
    residual = z0_on_z1 - z1_data["epsilon"]
    abs_residual = np.abs(residual)
    finite_residual = residual[mask]
    finite_abs = abs_residual[mask]
    finite_reference = np.abs(z1_data["epsilon"][mask])

    max_index = np.unravel_index(np.nanargmax(abs_residual), abs_residual.shape)
    thresholds = {}
    for level in DEFAULT_LEVELS:
        thresholds[f"{level:g}"] = {
            "area_frac_scaled_z0": float(np.mean(z0_on_z1[mask] <= level)),
            "area_frac_z1": float(np.mean(z1_data["epsilon"][mask] <= level)),
            "classification_disagree": float(
                np.mean(
                    (z0_on_z1[mask] <= level) != (z1_data["epsilon"][mask] <= level)
                )
            ),
        }

    return {
        "scaled_mcz": scaled_mcz,
        "z0_on_z1": z0_on_z1,
        "residual": residual,
        "abs_residual": abs_residual,
        "scale": scale,
        "summary": {
            "grid_match": bool(np.allclose(scaled_mcz, z1_data["mcz"])),
            "max_mass_grid_abs_diff": float(
                np.max(np.abs(scaled_mcz - z1_data["mcz"]))
            ),
            "points_compared": int(mask.sum()),
            "mean_abs_diff": float(np.mean(finite_abs)),
            "median_abs_diff": float(np.median(finite_abs)),
            "rms_diff": float(np.sqrt(np.mean(finite_residual**2))),
            "max_abs_diff": float(np.max(finite_abs)),
            "p95_abs_diff": float(np.percentile(finite_abs, 95)),
            "mean_relative_diff": float(
                np.mean(finite_abs / np.maximum(finite_reference, 1e-15))
            ),
            "corrcoef": float(
                np.corrcoef(z0_on_z1[mask], z1_data["epsilon"][mask])[0, 1]
            ),
            "max_residual_location": {
                "mcz": float(z1_data["mcz"][max_index[0]]),
                "td_ms": float(z1_data["td"][max_index[1]] * 1e3),
                "scaled_z0": float(z0_on_z1[max_index]),
                "z1": float(z1_data["epsilon"][max_index]),
            },
            "thresholds": thresholds,
        },
    }


def add_contours(
    ax: plt.Axes,
    mcz: np.ndarray,
    td: np.ndarray,
    epsilon: np.ndarray,
    levels: list[float],
    color: str,
) -> None:
    ax.contour(mcz, td * 1e3, epsilon.T, levels=levels, colors=color, linewidths=1.5)


def plot_comparison(
    z0_data: dict[str, np.ndarray],
    z1_data: dict[str, np.ndarray],
    comparison: dict,
    out_pdf: str,
    show: bool,
) -> None:
    z0_on_z1 = comparison["z0_on_z1"]
    residual = comparison["residual"]
    abs_residual = comparison["abs_residual"]
    summary = comparison["summary"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    extent = [
        z1_data["mcz"][0],
        z1_data["mcz"][-1],
        z1_data["td"][0] * 1e3,
        z1_data["td"][-1] * 1e3,
    ]

    image0 = axes[0, 0].imshow(
        z0_on_z1.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="jet",
    )
    axes[0, 0].set_title(r"Scaled z$\approx$0 mismatch on z=1 mass axis")
    axes[0, 0].set_ylabel(r"$\Delta t_d$ [ms]")
    fig.colorbar(image0, ax=axes[0, 0], label=r"minimum mismatch $\epsilon$")

    image1 = axes[0, 1].imshow(
        z1_data["epsilon"].T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="jet",
    )
    axes[0, 1].set_title(r"Stored z=1 mismatch")
    fig.colorbar(image1, ax=axes[0, 1], label=r"minimum mismatch $\epsilon$")

    vmax = float(np.nanmax(np.abs(residual)))
    image2 = axes[1, 0].imshow(
        residual.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    axes[1, 0].set_title("Residual: scaled z~0 minus z=1")
    axes[1, 0].set_xlabel(r"$\mathcal{M}_{cz}$ [$M_\odot$]")
    axes[1, 0].set_ylabel(r"$\Delta t_d$ [ms]")
    fig.colorbar(image2, ax=axes[1, 0], label=r"$\Delta \epsilon$")

    image3 = axes[1, 1].imshow(
        abs_residual.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="jet",
    )
    add_contours(
        axes[1, 1], z1_data["mcz"], z1_data["td"], z0_on_z1, DEFAULT_LEVELS, "white"
    )
    add_contours(
        axes[1, 1],
        z1_data["mcz"],
        z1_data["td"],
        z1_data["epsilon"],
        DEFAULT_LEVELS,
        "cyan",
    )
    axes[1, 1].set_title("Absolute residual with contour overlays")
    axes[1, 1].set_xlabel(r"$\mathcal{M}_{cz}$ [$M_\odot$]")
    axes[1, 1].legend(
        handles=[
            Line2D([0], [0], color="white", lw=1.5, label="scaled z~0"),
            Line2D([0], [0], color="cyan", lw=1.5, label="z=1"),
        ],
        loc="upper left",
    )
    fig.colorbar(image3, ax=axes[1, 1], label=r"$|\Delta \epsilon|$")

    fig.suptitle(
        (
            r"Edge-on mismatch scaling check: "
            r"$\mathcal{M}_{cz}(z\approx 0) \times (1+z_0)/(1+1)$ compared to stored z=1 grid"
            "\n"
            f"corr={summary['corrcoef']:.12f}, mean |delta|={summary['mean_abs_diff']:.3e}, max |delta|={summary['max_abs_diff']:.3e}"
        ),
        fontsize=14,
    )

    fig.savefig(out_pdf, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_residual_only(
    z1_data: dict[str, np.ndarray],
    comparison: dict,
    out_pdf: str,
    show: bool,
) -> None:
    residual = comparison["residual"]
    summary = comparison["summary"]
    extent = [
        z1_data["mcz"][0],
        z1_data["mcz"][-1],
        z1_data["td"][0] * 1e3,
        z1_data["td"][-1] * 1e3,
    ]
    vmax = float(np.nanmax(np.abs(residual)))

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    image = ax.imshow(
        residual.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_title("Residual: scaled z~0 minus z=1")
    ax.set_xlabel(r"$\mathcal{M}_{cz}$ [$M_\odot$]")
    ax.set_ylabel(r"$\Delta t_d$ [ms]")
    fig.colorbar(image, ax=ax, label=r"$\Delta \epsilon$")
    fig.suptitle(
        f"mean |delta|={summary['mean_abs_diff']:.3e}, max |delta|={summary['max_abs_diff']:.3e}",
        fontsize=12,
    )

    fig.savefig(out_pdf, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the mismatch-surface comparison between scaled z~0 and z=1 contour files."
    )
    parser.add_argument("--z0-path", default=DEFAULT_Z0_PATH)
    parser.add_argument("--z1-path", default=DEFAULT_Z1_PATH)
    parser.add_argument(
        "--z0",
        type=float,
        default=1e-6,
        help="Redshift associated with the first contour file.",
    )
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    z0_data = load_contour(args.z0_path)
    z1_data = load_contour(args.z1_path)
    comparison = compute_comparison(z0_data, z1_data, args.z0)

    out_pdf = os.path.join(args.outdir, f"{args.basename}.pdf")
    out_json = os.path.join(args.outdir, f"{args.basename}.json")
    residual_pdf = os.path.join(args.outdir, f"{args.basename}_residual.pdf")

    plot_comparison(z0_data, z1_data, comparison, out_pdf, args.show)
    plot_residual_only(z1_data, comparison, residual_pdf, args.show)

    report = {
        "z0_path": args.z0_path,
        "z1_path": args.z1_path,
        "z0": args.z0,
        "out_pdf": out_pdf,
        "residual_pdf": residual_pdf,
        **comparison["summary"],
    }
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    print(f"Saved plot: {out_pdf}")
    print(f"Saved plot: {residual_pdf}")
    print(f"Saved summary: {out_json}")
    print(json.dumps(comparison["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
