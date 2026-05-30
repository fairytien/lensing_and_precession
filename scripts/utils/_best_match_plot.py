"""Shared rendering helpers for plot_contour_*_from_best_match entry points."""

import os

import numpy as np
import matplotlib.pyplot as plt

from modules.plot_utils import save_figure, LBL_MIN_EPS_LP

VARIABLE_MAPPING = {
    "epsilon": {
        "dataset": "epsilon_min",
        "label": LBL_MIN_EPS_LP,
        "suffix": "epsilon_min",
    },
    "omega": {
        "dataset": "omega_best",
        "label": r"$\tilde{\Omega}_{\mathrm{best}}$",
        "suffix": "omega_best",
    },
    "theta": {
        "dataset": "theta_best",
        "label": r"$\tilde{\theta}_{\mathrm{best}}$",
        "suffix": "theta_best",
    },
}


def render_best_match_contour(
    x_arr,
    y_arr,
    Zmap,
    x_label,
    y_label,
    cbar_label,
    title,
    output_path,
    overlay_fn=None,
):
    """Render and save a best-match contour figure.

    Parameters
    ----------
    x_arr : 1-D array  (td in ms)
    y_arr : 1-D array  (mcz in M_sun, or I)
    Zmap  : 2-D array  shape (len(y_arr), len(x_arr))
    overlay_fn : optional callable(), called between the contour fill and
        tight_layout.  Use it to draw overlays or add a legend.
    """
    X, Y = np.meshgrid(x_arr, y_arr)
    plt.figure(figsize=(8, 6))
    cf = plt.contourf(X, Y, Zmap, levels=100, cmap="jet")
    cbar = plt.colorbar(cf)
    cbar.set_label(cbar_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    if overlay_fn is not None:
        overlay_fn()
    save_figure(plt.gcf(), output_path)


def build_figure_path(base_path, variable, has_overlays=False):
    """Return figure path with variable and overlay suffixes inserted before the extension."""
    var_info = VARIABLE_MAPPING[variable]
    suffixes = []
    if variable != "epsilon":
        suffixes.append(var_info["suffix"])
    if has_overlays:
        suffixes.append("overlayed")
    path_without_ext, ext = os.path.splitext(base_path)
    if suffixes:
        return f"{path_without_ext}_{'_'.join(suffixes)}{ext}"
    return base_path
