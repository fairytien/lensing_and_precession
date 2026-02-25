import os
import sys
from fractions import Fraction
from typing import Any, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.functions_v2 import *
from modules.plot_utils import *
from modules.cosmology import apply_z
from modules.filenames import append_z_tag_to_path

import numpy as np
import matplotlib.pyplot as plt


def make_frequency_array(f_min, f_cut, delta_f=None, npoints=None):
    if f_min < 0:
        raise ValueError("f_min must be non-negative")
    if f_cut <= f_min:
        raise ValueError("f_cut must be greater than f_min")
    if npoints is not None:
        if int(npoints) < 2:
            raise ValueError("npoints must be >= 2")
        return np.linspace(f_min, f_cut, int(npoints))
    if delta_f is None:
        raise ValueError("Provide either delta_f or npoints")
    if delta_f <= 0:
        raise ValueError("delta_f must be > 0")
    return np.arange(f_min, f_cut, delta_f)


def compute_phase(strain):
    return np.unwrap(np.angle(strain))


def _validate_amplitude_mode(amplitude_mode: str) -> None:
    if amplitude_mode not in {"abs", "ratio"}:
        raise ValueError("amplitude_mode must be either 'abs' or 'ratio'")


def _safe_relative_amplitude(numerator: Any, denominator: Any) -> np.ndarray:
    numerator_arr = np.asarray(numerator)
    denominator_arr = np.asarray(denominator)
    den_abs = np.abs(denominator_arr)
    out = np.full_like(den_abs, np.nan, dtype=float)
    np.divide(np.abs(numerator_arr), den_abs, out=out, where=den_abs > 0)
    return out - 1.0


def _amplitude_yvals(
    lensed_or_rp: Any,
    unlensed_or_np: Any,
    amplitude_mode: str,
) -> np.ndarray:
    if amplitude_mode == "abs":
        return np.abs(np.asarray(lensed_or_rp))
    return _safe_relative_amplitude(lensed_or_rp, unlensed_or_np)


def _plot_baseline(
    ax_amp,
    ax_phase,
    f_arr: np.ndarray,
    baseline_amp: np.ndarray,
    amplitude_mode: str,
    baseline_color: str,
    label: str,
) -> None:
    amp_ref = baseline_amp if amplitude_mode == "abs" else np.zeros_like(f_arr)
    ax_amp.plot(
        f_arr,
        amp_ref,
        ls="-",
        color=baseline_color,
        label=label,
    )
    ax_phase.plot(
        f_arr,
        np.zeros_like(f_arr),
        ls="-",
        color=baseline_color,
        label=label,
    )


def _format_angle_pi(angle, max_den=24):
    r"""Return a TeX-ready string for angle in radians as multiple of \pi.
    Examples: 0 -> 0, \pi/3 -> '$\pi/3$', 2\pi/3 -> '$2\pi/3$', \pi -> '$\pi$', 3\pi/2 -> '$3\pi/2$'.
    Returns a string already wrapped in math mode $...$.
    """
    x = angle / np.pi
    if np.isclose(x, 0.0, atol=1e-10):
        return "0"
    frac = Fraction(x).limit_denominator(max_den)
    n, d = frac.numerator, frac.denominator
    sign = "-" if n < 0 else ""
    n_abs = abs(n)
    if d == 1:
        # integer multiple
        return rf"${sign}\pi$" if n_abs == 1 else rf"${sign}{n_abs}\pi$"
    # general fraction
    return rf"${sign}\pi/{d}$" if n_abs == 1 else rf"${sign}{n_abs}\pi/{d}$"


def customize_2x2_axes_ratio(axes):
    """Like modules.plot_utils.customize_2x2_axes but for ratio panels:
    - Do NOT use log yscale
    - Do NOT force the same y-limits across rows
    Keeps legends, grids, labels, and tick sizes consistent.
    """
    # top panel
    axes[0, 0].legend(
        bbox_to_anchor=(2.3, 1), loc="upper left", borderaxespad=0.0, fontsize=20
    )
    axes[0, 0].tick_params(axis="both", which="major", labelsize=18)
    axes[0, 0].grid()
    axes[0, 1].tick_params(axis="both", which="major", labelsize=18)
    axes[0, 1].grid()

    # bottom panel
    axes[1, 0].legend(
        bbox_to_anchor=(2.3, 1), loc="upper left", borderaxespad=0.0, fontsize=20
    )
    axes[1, 0].set_xlabel("f (Hz)", fontsize=24)
    axes[1, 0].tick_params(axis="both", which="major", labelsize=18)
    axes[1, 0].grid()
    axes[1, 1].set_xlabel("f (Hz)", fontsize=24)
    axes[1, 1].tick_params(axis="both", which="major", labelsize=18)
    axes[1, 1].grid()

    # Ensure linear y-scale for amplitude panels and autoscale independently
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_yscale("linear")
        ax.relim()
        ax.autoscale_view()


def customize_3x2_axes_abs(axes):
    """Customizer for 3x2 figures (abs mode):
    - Log scale on amplitude panels (col 0)
    - Legends on each amplitude row
    - Grids and tick sizes consistent
    """
    for r in range(3):
        axes[r, 0].legend(
            bbox_to_anchor=(2.3, 1), loc="upper left", borderaxespad=0.0, fontsize=20
        )
        axes[r, 0].tick_params(axis="both", which="major", labelsize=18)
        axes[r, 0].grid()
        axes[r, 0].set_yscale("log")
        axes[r, 1].tick_params(axis="both", which="major", labelsize=18)
        axes[r, 1].grid()
    # x labels on bottom row
    axes[2, 0].set_xlabel("f (Hz)", fontsize=24)
    axes[2, 1].set_xlabel("f (Hz)", fontsize=24)


def customize_3x2_axes_ratio(axes):
    """Customizer for 3x2 figures (ratio mode):
    - Linear y-scale on amplitude panels (col 0)
    - Legends on each amplitude row
    - Grids and tick sizes consistent
    """
    for r in range(3):
        axes[r, 0].legend(
            bbox_to_anchor=(2.3, 1), loc="upper left", borderaxespad=0.0, fontsize=20
        )
        axes[r, 0].tick_params(axis="both", which="major", labelsize=18)
        axes[r, 0].grid()
        axes[r, 0].set_yscale("linear")
        axes[r, 1].tick_params(axis="both", which="major", labelsize=18)
        axes[r, 1].grid()
    # x labels on bottom row
    axes[2, 0].set_xlabel("f (Hz)", fontsize=24)
    axes[2, 1].set_xlabel("f (Hz)", fontsize=24)


def plot_lensing_figure(
    lens_params_base,
    td_arr,
    def_y,
    I_arr,
    def_td,
    *,
    amplitude_mode="abs",
    line_styles: Optional[Sequence[str]] = None,
    baseline_color="darkorange",
    f_min=20,
    npoints=10000,
    z: Optional[float] = None,
    save_path: Optional[str] = None,
    bbox_inches="tight",
    pad_inches=0.02,
):
    """Plot 2x2 lensing panels varying (top) time delay via MLz and (bottom) I via y.

    amplitude_mode: "abs" for |h| vs NP, "ratio" for (|h_L|/|h_NP|)-1
    save_path: optional file path to save the figure; format inferred by extension.
    Returns (fig, axes).
    """
    _validate_amplitude_mode(amplitude_mode)
    if npoints < 2:
        raise ValueError("npoints must be >= 2")
    if line_styles is None:
        line_styles = ["-", "--", ":"]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
    fig.subplots_adjust(wspace=0.25)

    # --- Top row: vary time delay (via MLz) at fixed impact y ---
    for i, td in enumerate(np.atleast_1d(td_arr)):
        lens_params = lens_params_base.copy()
        MLz = get_MLz_from_td(td, def_y)
        lens_params["MLz"] = MLz * solar_mass
        if z is not None:
            lens_params = apply_z(lens_params, z)
        lens_inst = LensingGeo(lens_params)
        f_cut = lens_inst.f_cut()
        f_arr = make_frequency_array(f_min, f_cut, npoints=npoints)
        unlensed_strain = lens_inst.hI(f_arr)
        lensed_strain = lens_inst.strain(f_arr)
        phase_diff = compute_phase(lensed_strain) - compute_phase(unlensed_strain)
        yvals = _amplitude_yvals(lensed_strain, unlensed_strain, amplitude_mode)
        if i == 0:
            _plot_baseline(
                axes[0, 0],
                axes[0, 1],
                f_arr,
                np.abs(unlensed_strain),
                amplitude_mode,
                baseline_color,
                "Unlensed",
            )
        Delta_td = lens_inst.td()
        axes[0, 0].plot(
            f_arr,
            yvals,
            ls=line_styles[i % len(line_styles)],
            color="black",
            label=rf"$\Delta t_d$ = {Delta_td*1000:.2g} ms",
        )
        axes[0, 1].plot(
            f_arr,
            phase_diff,
            ls=line_styles[i % len(line_styles)],
            color="black",
            label=rf"$\Delta t_d$ = {Delta_td*1000:.2g} ms",
        )

    # --- Bottom row: vary I (via y) at fixed time delay def_td ---
    for i, I in enumerate(np.atleast_1d(I_arr)):
        lens_params = lens_params_base.copy()
        y = get_y_from_I(I)
        MLz = get_MLz_from_td(def_td, y)
        lens_params["y"] = y
        lens_params["MLz"] = MLz * solar_mass
        if z is not None:
            lens_params = apply_z(lens_params, z)
        lens_inst = LensingGeo(lens_params)
        f_cut = lens_inst.f_cut()
        f_arr = make_frequency_array(f_min, f_cut, npoints=npoints)
        unlensed_strain = lens_inst.hI(f_arr)
        lensed_strain = lens_inst.strain(f_arr)
        phase_diff = compute_phase(lensed_strain) - compute_phase(unlensed_strain)
        yvals = _amplitude_yvals(lensed_strain, unlensed_strain, amplitude_mode)
        if i == 0:
            _plot_baseline(
                axes[1, 0],
                axes[1, 1],
                f_arr,
                np.abs(unlensed_strain),
                amplitude_mode,
                baseline_color,
                "Unlensed",
            )
        I_val = lens_inst.I()
        axes[1, 0].plot(
            f_arr,
            yvals,
            ls=line_styles[i % len(line_styles)],
            color="black",
            label=rf"$I$ = {I_val:.2g}",
        )
        axes[1, 1].plot(
            f_arr,
            phase_diff,
            ls=line_styles[i % len(line_styles)],
            color="black",
            label=rf"$I$ = {I_val:.2g}",
        )

    # Labels and styling
    if amplitude_mode == "abs":
        fig.text(0.06, 0.5, r"$|\~{h}|$", va="center", rotation="vertical", fontsize=24)
        fig.text(
            0.49,
            0.5,
            r"$\Phi_L - \Phi_{NP}$ (rad)",
            va="center",
            rotation="vertical",
            fontsize=24,
        )
        customize_2x2_axes(axes)
    else:
        fig.text(
            0.06,
            0.5,
            r"$\left( B_L/B_{NP} \right) - 1$",
            va="center",
            rotation="vertical",
            fontsize=24,
        )
        fig.text(
            0.49,
            0.5,
            r"$\Phi_L - \Phi_{NP}$ (rad)",
            va="center",
            rotation="vertical",
            fontsize=24,
        )
        customize_2x2_axes_ratio(axes)

    if z is not None:
        print(f"Applied redshift z={z} (mcz treated as source-frame)")

    # Optional save (format inferred from extension)
    if save_path:
        out_path = append_z_tag_to_path(save_path, z)
        if out_path is None:
            raise ValueError("append_z_tag_to_path returned None for save_path")
        fig.savefig(
            out_path,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
        )
    return fig, axes


def plot_precessing_figure(
    RP_params_1,
    NP_params_1,
    *,
    theta_vals,
    omega_vals,
    gamma_vals=(0.0, np.pi / 3, 2 * np.pi / 3),
    fixed_theta=4,
    fixed_omega=2,
    amplitude_mode="abs",
    line_styles: Optional[Sequence[str]] = None,
    baseline_color="darkorange",
    f_min=20,
    delta_f=0.05,
    z: Optional[float] = None,
    save_path: Optional[str] = None,
    bbox_inches="tight",
    pad_inches=0.02,
):
    """Plot 3x2 precessing panels:
    - Row 0: vary omega_tilde (fixed theta)
    - Row 1: vary theta_tilde (fixed omega)
    - Row 2: vary gamma_P (fixed theta and omega)
    amplitude_mode: "abs" or "ratio".
    save_path: optional file path to save the figure; format inferred by extension.
    Returns (fig, axes).
    """
    _validate_amplitude_mode(amplitude_mode)
    if delta_f <= 0:
        raise ValueError("delta_f must be > 0")
    if line_styles is None:
        line_styles = ["-", "--", ":"]
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 18))
    fig.subplots_adjust(wspace=0.25)

    # Helper to run a sweep
    def sweep(row, vary_key, values, fixed_items, label_tex):
        for i, val in enumerate(values):
            RP_params = RP_params_1.copy()
            NP_params = NP_params_1.copy()
            # apply fixed
            for k, v in fixed_items.items():
                RP_params[k] = v
            RP_params[vary_key] = val
            if z is not None:
                RP_params = apply_z(RP_params, z)
                NP_params = apply_z(NP_params, z)
            RP_inst = Precessing(RP_params)
            f_cut = RP_inst.f_cut()
            f_arr = make_frequency_array(f_min, f_cut, delta_f=delta_f)
            RP_strain = RP_inst.strain(f_arr)
            NP_inst = Precessing(NP_params)
            NP_strain = NP_inst.strain(f_arr)
            phase_diff = compute_phase(RP_strain) - compute_phase(NP_strain)
            yvals = _amplitude_yvals(RP_strain, NP_strain, amplitude_mode)
            if i == 0:
                _plot_baseline(
                    axes[row, 0],
                    axes[row, 1],
                    f_arr,
                    np.abs(NP_strain),
                    amplitude_mode,
                    baseline_color,
                    "NP",
                )
            if vary_key == "gamma_P":
                gamma_tex = _format_angle_pi(val)
                label_str = r"$\gamma_P$ = " + gamma_tex
            else:
                label_str = f"{label_tex} = {val}"
            axes[row, 0].plot(
                f_arr,
                yvals,
                label=label_str,
                color="black",
                ls=line_styles[i % len(line_styles)],
            )
            axes[row, 1].plot(
                f_arr,
                phase_diff,
                label=label_str,
                color="black",
                ls=line_styles[i % len(line_styles)],
            )

    # Row order: omega, theta, gamma
    sweep(
        0,
        "omega_tilde",
        np.atleast_1d(omega_vals),
        {"theta_tilde": fixed_theta},
        r"$\tilde{\Omega}$",
    )
    sweep(
        1,
        "theta_tilde",
        np.atleast_1d(theta_vals),
        {"omega_tilde": fixed_omega},
        r"$\tilde{\theta}$",
    )
    sweep(
        2,
        "gamma_P",
        np.atleast_1d(gamma_vals),
        {"theta_tilde": fixed_theta, "omega_tilde": fixed_omega},
        r"$\gamma_P$",
    )

    # Labels and styling
    if amplitude_mode == "abs":
        fig.text(0.06, 0.5, r"$|\~{h}|$", va="center", rotation="vertical", fontsize=24)
        fig.text(
            0.49,
            0.5,
            r"$\Phi_{RP} - \Phi_{NP}$ (rad)",
            va="center",
            rotation="vertical",
            fontsize=24,
        )
        customize_3x2_axes_abs(axes)
    else:
        fig.text(
            0.06,
            0.5,
            r"$\left( B_{RP}/B_{NP} \right) - 1$",
            va="center",
            rotation="vertical",
            fontsize=24,
        )
        fig.text(
            0.49,
            0.5,
            r"$\Phi_{RP} - \Phi_{NP}$ (rad)",
            va="center",
            rotation="vertical",
            fontsize=24,
        )
        customize_3x2_axes_ratio(axes)

    if z is not None:
        print(f"Applied redshift z={z} (mcz treated as source-frame)")

    # Optional save (format inferred from extension)
    if save_path:
        out_path = append_z_tag_to_path(save_path, z)
        if out_path is None:
            raise ValueError("append_z_tag_to_path returned None for save_path")
        fig.savefig(
            out_path,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
        )
    return fig, axes
