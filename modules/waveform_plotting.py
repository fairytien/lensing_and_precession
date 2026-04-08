import os
import sys
from typing import Any, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.Classes_v3 import LensingGeo, Precessing
from modules.default_params_v3 import SOLMASS2SEC
from modules.functions_v3 import get_MLz_from_td, get_y_from_I
from modules.cosmology import apply_z
from modules.filenames import _format_min_precision
from modules.plot_utils_v3 import (
    angle_to_pi_string,
    customize_2x1_axes_ratio,
    customize_2x2_axes,
    customize_2x2_axes_ratio,
    customize_3x2_axes_abs,
    customize_3x2_axes_ratio,
)


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


def plot_lensing_figure(
    lens_params_base,
    td_arr,
    def_y,
    I_arr,
    def_td,
    *,
    amplitude_mode="abs",
    line_styles: Sequence[str] = ("-", "--", ":"),
    line_colors: Sequence[str] = ("black",),
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
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
    fig.subplots_adjust(wspace=0.25)

    # --- Top row: vary time delay (via MLz) at fixed impact y ---
    for i, td in enumerate(np.atleast_1d(td_arr)):
        lens_params = lens_params_base.copy()
        MLz = get_MLz_from_td(td, def_y)
        lens_params["MLz"] = MLz * SOLMASS2SEC
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
            color=line_colors[i % len(line_colors)],
            label=rf"$\Delta t_d$ = {Delta_td*1000:.2g} ms",
        )
        axes[0, 1].plot(
            f_arr,
            phase_diff,
            ls=line_styles[i % len(line_styles)],
            color=line_colors[i % len(line_colors)],
            label=rf"$\Delta t_d$ = {Delta_td*1000:.2g} ms",
        )

    # --- Bottom row: vary I (via y) at fixed time delay def_td ---
    for i, I in enumerate(np.atleast_1d(I_arr)):
        lens_params = lens_params_base.copy()
        y = get_y_from_I(I)
        MLz = get_MLz_from_td(def_td, y)
        lens_params["y"] = y
        lens_params["MLz"] = MLz * SOLMASS2SEC
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
            color=line_colors[i % len(line_colors)],
            label=rf"$I$ = {I_val:.2g}",
        )
        axes[1, 1].plot(
            f_arr,
            phase_diff,
            ls=line_styles[i % len(line_styles)],
            color=line_colors[i % len(line_colors)],
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

    if save_path:
        mcz_msun = float(lens_params_base["mcz"]) / SOLMASS2SEC
        root, ext = os.path.splitext(save_path)
        out_path = (
            f"{root}{_format_min_precision(z, prefix='_z')}"
            f"_mcz{_format_min_precision(mcz_msun, suffix='Msun')}{ext}"
        )
        fig.savefig(
            str(out_path),
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
    line_styles: Sequence[str] = ("-", "--", ":"),
    line_colors: Sequence[str] = ("black",),
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
                gamma_tex = angle_to_pi_string(val, denom_thres=24)
                label_str = r"$\gamma_P$ = " + gamma_tex
            else:
                label_str = f"{label_tex} = {val}"
            axes[row, 0].plot(
                f_arr,
                yvals,
                label=label_str,
                color=line_colors[i % len(line_colors)],
                ls=line_styles[i % len(line_styles)],
            )
            axes[row, 1].plot(
                f_arr,
                phase_diff,
                label=label_str,
                color=line_colors[i % len(line_colors)],
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

    if save_path:
        mcz_msun = float(RP_params_1["mcz"]) / SOLMASS2SEC
        root, ext = os.path.splitext(save_path)
        out_path = (
            f"{root}{_format_min_precision(z, prefix='_z')}"
            f"_mcz{_format_min_precision(mcz_msun, suffix='Msun')}{ext}"
        )
        fig.savefig(
            str(out_path),
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
        )
    return fig, axes


def get_best_match_template_params(contour_data: dict) -> dict:
    """Return template params set to the global min-mismatch location.

    Prefers `contour_data["stats"]` keys when present; otherwise uses argmin on
    `epsilon_matrix` and corresponding entries in omega/theta/gamma matrices.
    """
    template_params = contour_data["template_params"].copy()
    stats = contour_data.get("stats", None)

    if isinstance(stats, dict) and all(
        key in stats
        for key in ("ep_min_omega_tilde", "ep_min_theta_tilde", "ep_min_gammaP")
    ):
        template_params["omega_tilde"] = float(stats["ep_min_omega_tilde"])
        template_params["theta_tilde"] = float(stats["ep_min_theta_tilde"])
        template_params["gamma_P"] = float(stats["ep_min_gammaP"])
        return template_params

    epsilon_matrix = np.asarray(contour_data["epsilon_matrix"], dtype=float)
    min_index = np.nanargmin(epsilon_matrix)
    row, col = np.unravel_index(min_index, epsilon_matrix.shape)

    template_params["omega_tilde"] = float(contour_data["omega_matrix"][row, col])
    template_params["theta_tilde"] = float(contour_data["theta_matrix"][row, col])
    template_params["gamma_P"] = float(contour_data["gammaP_min_matrix"][row, col])
    return template_params


def plot_best_match_overlay_from_contour(
    contour_data: dict,
    axes,
    *,
    f_min: float = 20.0,
    npoints: int = 10000,
    baseline_color: str = "darkorange",
    lensed_color: str = "magenta",
    np_linestyle: str = "--",
    np_label: str = "NP",
    rp_color: str = "black",
    rp_linestyle: str = "-",
    rp_label: str = "RP (best)",
) -> dict:
    """Plot lensed vs best-match RP overlay from one contour dictionary.

    Left panel: fractional amplitude change with template waveforms in the
    numerator, relative to the lensed source waveform (source/template form).
    Right panel: phase differences with templates on the left, i.e.
    ``Phi_t - Phi_s``.

    Returns summary metadata dict containing best-match coordinates and epsilon.
    """
    source_params = contour_data["source_params"].copy()
    template_params = get_best_match_template_params(contour_data)

    lensed_inst = LensingGeo(source_params)
    rp_inst = Precessing(template_params)
    f_cut = min(lensed_inst.f_cut(), rp_inst.f_cut())
    f_arr = make_frequency_array(f_min, f_cut, npoints=npoints)

    unlensed_source = lensed_inst.hI(f_arr)
    lensed_strain = np.asarray(lensed_inst.strain(f_arr, frequencySeries=False))
    rp_strain = np.asarray(rp_inst.strain(f_arr, frequencySeries=False))

    frac_np = _safe_relative_amplitude(unlensed_source, lensed_strain)
    frac_rp = _safe_relative_amplitude(rp_strain, lensed_strain)

    phase_lensed = compute_phase(lensed_strain)
    phase_np = compute_phase(unlensed_source) - phase_lensed
    phase_rp = compute_phase(rp_strain) - phase_lensed

    axes[0].plot(f_arr, np.zeros_like(f_arr), c=lensed_color, ls="-", label="lensed")
    axes[0].plot(f_arr, frac_np, c=baseline_color, ls=np_linestyle, label=np_label)
    axes[0].plot(f_arr, frac_rp, c=rp_color, ls=rp_linestyle, label=rp_label)

    axes[1].plot(f_arr, phase_np, c=baseline_color, ls=np_linestyle, label=np_label)
    axes[1].plot(f_arr, phase_rp, c=rp_color, ls=rp_linestyle, label=rp_label)

    epsilon_matrix = np.asarray(contour_data["epsilon_matrix"], dtype=float)
    best_epsilon = float(np.nanmin(epsilon_matrix))

    mcz_msun = float(contour_data.get("mcz_msun", np.nan))
    if not np.isfinite(mcz_msun):
        mcz_msun = float(source_params.get("mcz", np.nan)) / SOLMASS2SEC

    td_ms = float(contour_data.get("td_ms", np.nan))
    if not np.isfinite(td_ms):
        td_ms = float(lensed_inst.td() * 1e3)

    I_val = float(contour_data.get("I", np.nan))
    if not np.isfinite(I_val):
        I_val = float(lensed_inst.I())

    return {
        "omega_tilde": float(template_params["omega_tilde"]),
        "theta_tilde": float(template_params["theta_tilde"]),
        "gamma_P": float(template_params["gamma_P"]),
        "epsilon": best_epsilon,
        "mcz_msun": mcz_msun,
        "td_ms": td_ms,
        "I": I_val,
    }
