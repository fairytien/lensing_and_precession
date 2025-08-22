import sys, os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_pickle_data(filepath):
    """Load pickle data and return the loaded object"""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def create_comparison_contours():
    """Create comparison contours with same color scaling"""

    # Load both datasets
    v2_data = load_pickle_data(
        "data/v2_indiv_mismatch_L_RP_mcz20_td22ms_I0.6_thetaS1.047_phiS0.785_thetaJ0.524_phiJ1.047_2_2025-08-22_13-08-25.pkl"
    )
    v3_data = load_pickle_data(
        "data/v3_indiv_mismatch_L_RP_mcz20_td22ms_I0.6_thetaS1.047_phiS0.785_thetaJ0.524_phiJ1.047_3_2025-08-22_11-42-35.pkl"
    )

    # Extract matrices
    v2_omega = v2_data["omega_matrix"]
    v2_theta = v2_data["theta_matrix"]
    v2_epsilon = v2_data["epsilon_matrix"]

    v3_omega = v3_data["omega_matrix"]
    v3_theta = v3_data["theta_matrix"]
    v3_epsilon = v3_data["epsilon_matrix"]

    # Find global min/max for consistent color scaling
    global_min = min(v2_epsilon.min(), v3_epsilon.min())
    global_max = max(v2_epsilon.max(), v3_epsilon.max())

    print(f"Global epsilon range: {global_min:.6f} to {global_max:.6f}")
    print(f"v2 epsilon range: {v2_epsilon.min():.6f} to {v2_epsilon.max():.6f}")
    print(f"v3 epsilon range: {v3_epsilon.min():.6f} to {v3_epsilon.max():.6f}")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot v2 contour
    cf1 = ax1.contourf(
        v2_omega,
        v2_theta,
        v2_epsilon,
        levels=np.linspace(global_min, global_max, 100),
        cmap="jet",
        extend="both",
    )
    ax1.set_title("v2: Lensing vs Regular Precession")
    ax1.set_xlabel(r"$\tilde{\Omega}$")
    ax1.set_ylabel(r"$\tilde{\theta}$")

    # Plot v3 contour
    cf2 = ax2.contourf(
        v3_omega,
        v3_theta,
        v3_epsilon,
        levels=np.linspace(global_min, global_max, 100),
        cmap="jet",
        extend="both",
    )
    ax2.set_title("v3: Lensing vs Regular Precession")
    ax2.set_xlabel(r"$\tilde{\Omega}$")
    ax2.set_ylabel(r"$\tilde{\theta}$")

    # Add colorbar with consistent scaling
    cbar = plt.colorbar(cf2, ax=[ax1, ax2], shrink=0.8)
    cbar.set_label(r"$\epsilon(\tilde{h}_{\mathrm{L}}, \tilde{h}_{\mathrm{RP}})$")

    plt.tight_layout()

    # Save figure
    fig_path = "figures/v2_v3_comparison_same_scale.pdf"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Comparison figure saved as {fig_path}")

    # Also create individual plots with same scaling for reference
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))

    # Individual v2 plot
    cf3 = ax3.contourf(
        v2_omega,
        v2_theta,
        v2_epsilon,
        levels=np.linspace(global_min, global_max, 100),
        cmap="jet",
        extend="both",
    )
    ax3.set_title("v2: Lensing vs Regular Precession (Same Scale)")
    ax3.set_xlabel(r"$\tilde{\Omega}$")
    ax3.set_ylabel(r"$\tilde{\theta}$")
    cbar2 = plt.colorbar(cf3, ax=ax3)
    cbar2.set_label(r"$\epsilon(\tilde{h}_{\mathrm{L}}, \tilde{h}_{\mathrm{RP}})$")

    # Individual v3 plot
    cf4 = ax4.contourf(
        v3_omega,
        v3_theta,
        v3_epsilon,
        levels=np.linspace(global_min, global_max, 100),
        cmap="jet",
        extend="both",
    )
    ax4.set_title("v3: Lensing vs Regular Precession (Same Scale)")
    ax4.set_xlabel(r"$\tilde{\Omega}$")
    ax4.set_ylabel(r"$\tilde{\theta}$")
    cbar3 = plt.colorbar(cf4, ax=ax4)
    cbar3.set_label(r"$\epsilon(\tilde{h}_{\mathrm{L}}, \tilde{h}_{\mathrm{RP}})$")

    plt.tight_layout()

    # Save individual plots
    fig2_path = "figures/v2_v3_individual_same_scale.pdf"
    plt.savefig(fig2_path, dpi=200, bbox_inches="tight")
    print(f"Individual plots saved as {fig2_path}")

    # Print some statistics for comparison
    print("\nGrid Information:")
    print(f"v2 grid shape: {v2_omega.shape}")
    print(f"v3 grid shape: {v3_omega.shape}")
    print(f"v2 omega range: {v2_omega.min():.3f} to {v2_omega.max():.3f}")
    print(f"v3 omega range: {v3_omega.min():.3f} to {v3_omega.max():.3f}")
    print(f"v2 theta range: {v2_theta.min():.3f} to {v2_theta.max():.3f}")
    print(f"v3 theta range: {v3_theta.min():.3f} to {v3_theta.max():.3f}")

    plt.show()


if __name__ == "__main__":
    create_comparison_contours()
