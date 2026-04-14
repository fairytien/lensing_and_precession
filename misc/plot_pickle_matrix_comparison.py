import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

FILE_A = "data/v3_indiv_contour_mcz20_td30ms_I0.5_thetaS0.785_phiS0.0_thetaJ1.571_phiJ1.571_z0.28_2026-03-03_06-04-50.pkl"
FILE_B = "data/v3_indiv_contour_mcz20_td30ms_I0.5_thetaS0.785_phiS0.0_thetaJ1.571_phiJ1.571_2026-03-03_06-06-17.pkl"
OUT_DIR = "figures/contour_omega_theta"
OUT_FILE = "pickle_matrix_comparison_mcz20_td30ms_2026-03-03.png"


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def matrix_stats(a, b):
    diff = b - a
    abs_diff = np.abs(diff)
    finite = np.isfinite(a) & np.isfinite(b)

    corr = np.nan
    if np.any(finite):
        af = a[finite].ravel()
        bf = b[finite].ravel()
        if af.size > 1 and np.nanstd(af) > 0 and np.nanstd(bf) > 0:
            corr = np.corrcoef(af, bf)[0, 1]

    return {
        "shape": a.shape,
        "max_abs_diff": float(np.nanmax(abs_diff)),
        "mean_abs_diff": float(np.nanmean(abs_diff)),
        "mean_diff": float(np.nanmean(diff)),
        "corr": float(corr),
    }


def draw_triplet(ax_row, a, b, label):
    diff = b - a
    vmin = np.nanmin([np.nanmin(a), np.nanmin(b)])
    vmax = np.nanmax([np.nanmax(a), np.nanmax(b)])
    dmax = np.nanmax(np.abs(diff))

    im0 = ax_row[0].imshow(
        a, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis"
    )
    ax_row[0].set_title(f"{label} (A)")
    plt.colorbar(im0, ax=ax_row[0], fraction=0.046, pad=0.04)

    im1 = ax_row[1].imshow(
        b, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis"
    )
    ax_row[1].set_title(f"{label} (B)")
    plt.colorbar(im1, ax=ax_row[1], fraction=0.046, pad=0.04)

    im2 = ax_row[2].imshow(
        diff, origin="lower", aspect="auto", vmin=-dmax, vmax=dmax, cmap="RdBu_r"
    )
    ax_row[2].set_title(f"{label} (B - A)")
    plt.colorbar(im2, ax=ax_row[2], fraction=0.046, pad=0.04)

    for ax in ax_row:
        ax.set_xlabel("Column index")
        ax.set_ylabel("Row index")


def main():
    data_a = load_pickle(FILE_A)
    data_b = load_pickle(FILE_B)

    epsilon_a = np.asarray(data_a["epsilon_matrix"], dtype=float)
    epsilon_b = np.asarray(data_b["epsilon_matrix"], dtype=float)
    gamma_a = np.asarray(data_a["gammaP_min_matrix"], dtype=float)
    gamma_b = np.asarray(data_b["gammaP_min_matrix"], dtype=float)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, OUT_FILE)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    draw_triplet(axes[0], epsilon_a, epsilon_b, "epsilon_matrix")
    draw_triplet(axes[1], gamma_a, gamma_b, "gammaP_min_matrix")

    fig.suptitle("Pickle matrix comparison: A=z0.28 file, B=no-z file", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    eps_stats = matrix_stats(epsilon_a, epsilon_b)
    gam_stats = matrix_stats(gamma_a, gamma_b)

    print("Saved figure:", out_path)
    print("epsilon_matrix stats:", eps_stats)
    print("gammaP_min_matrix stats:", gam_stats)


if __name__ == "__main__":
    main()
