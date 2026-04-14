import os
import sys
from typing import Any, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.Classes import LensingGeo, Precessing
from modules.default_params import SOLMASS2SEC
from modules.functions import get_MLz_from_td, get_y_from_I
from modules.cosmology import apply_z
from modules.filenames import _format_min_precision
from modules.plot_utils import (
    apply_physics_paper_style,
    angle_to_pi_string,
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


def _apply_optional_redshift(params: dict, z: Optional[float]) -> dict:
    if z is None:
        return params
    return apply_z(params, z)


def _build_lensing_params_from_td(
    lens_params_base: dict,
    td: float,
    y: float,
    z: Optional[float],
) -> dict:
    lens_params = lens_params_base.copy()
    MLz = get_MLz_from_td(td, y)
    lens_params["MLz"] = MLz * SOLMASS2SEC
    return _apply_optional_redshift(lens_params, z)


def _build_lensing_params_from_I(
    lens_params_base: dict,
    I: float,
    td: float,
    z: Optional[float],
) -> dict:
    lens_params = lens_params_base.copy()
    y = get_y_from_I(I)
    MLz = get_MLz_from_td(td, y)
    lens_params["y"] = y
    lens_params["MLz"] = MLz * SOLMASS2SEC
    return _apply_optional_redshift(lens_params, z)


def _build_precessing_instances(
    RP_params_base: dict,
    NP_params_base: dict,
    vary_key: str,
    val: float,
    fixed_items: dict,
    z: Optional[float],
):
    RP_params = RP_params_base.copy()
    for k, v in fixed_items.items():
        RP_params[k] = v
    RP_params[vary_key] = val

    RP_params = _apply_optional_redshift(RP_params, z)
    NP_params = _apply_optional_redshift(NP_params_base.copy(), z)
    return Precessing(RP_params), Precessing(NP_params)


def _add_vertical_axis_labels(
    fig,
    axes,
    amp_label: str,
    phase_label: str,
    *,
    left_label_x: float = 0.06,
    reference_row: int = 0,
    fontsize: int = 24,
    phase_label_x_shift: float = 0.0,
) -> None:
    left_axis_x = axes[reference_row, 0].get_position().x0
    right_axis_x = axes[reference_row, 1].get_position().x0
    label_gap = left_axis_x - left_label_x
    phase_label_x = float(
        np.clip(right_axis_x - label_gap + phase_label_x_shift, 0.0, 1.0)
    )

    fig.text(
        left_label_x,
        0.5,
        amp_label,
        va="center",
        rotation="vertical",
        fontsize=fontsize,
    )
    fig.text(
        phase_label_x,
        0.5,
        phase_label,
        va="center",
        rotation="vertical",
        fontsize=fontsize,
    )


def _build_save_path_with_mcz(
    save_path: str,
    z: Optional[float],
    mcz_sec: float,
) -> str:
    mcz_msun = float(mcz_sec) / SOLMASS2SEC
    root, ext = os.path.splitext(save_path)
    return (
        f"{root}{_format_min_precision(z, prefix='_z')}"
        f"_mcz{_format_min_precision(mcz_msun, suffix='Msun')}{ext}"
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
    share_frequency_range: bool = True,
    z: Optional[float] = None,
    save_path: Optional[str] = None,
    bbox_inches="tight",
    pad_inches=0.02,
):
    """Plot 2x2 lensing panels varying (top) time delay via MLz and (bottom) I via y.

    amplitude_mode: "abs" for |h| vs NP, "ratio" for (|h_L|/|h_NP|)-1
    share_frequency_range: if True, all panels use a common f range [f_min, f_cut_common]
        for cleaner paper-ready comparison across subplots.
    save_path: optional file path to save the figure; format inferred by extension.
    Returns (fig, axes).
    """
    apply_physics_paper_style(base_font=12, label_font=14, tick_font=11, legend_font=11)
    _validate_amplitude_mode(amplitude_mode)
    if npoints < 2:
        raise ValueError("npoints must be >= 2")
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12), sharex=True)
    fig.subplots_adjust(wspace=0.25, hspace=0.12)

    f_cut_common = None
    if share_frequency_range:
        f_cut_candidates = []

        for td in np.atleast_1d(td_arr):
            lens_params = _build_lensing_params_from_td(
                lens_params_base,
                td,
                def_y,
                z,
            )
            f_cut_candidates.append(float(LensingGeo(lens_params).f_cut()))

        for I in np.atleast_1d(I_arr):
            lens_params = _build_lensing_params_from_I(
                lens_params_base,
                I,
                def_td,
                z,
            )
            f_cut_candidates.append(float(LensingGeo(lens_params).f_cut()))

        f_cut_common = float(np.min(f_cut_candidates))
        if f_cut_common <= f_min:
            raise ValueError("Common f_cut must be greater than f_min")

    # --- Top row: vary time delay (via MLz) at fixed impact y ---
    for i, td in enumerate(np.atleast_1d(td_arr)):
        lens_params = _build_lensing_params_from_td(
            lens_params_base,
            td,
            def_y,
            z,
        )
        lens_inst = LensingGeo(lens_params)
        f_cut = f_cut_common if share_frequency_range else lens_inst.f_cut()
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
            label=rf"$\Delta t_{{\mathrm{{d}}}} = {Delta_td*1000:.2g}\,\mathrm{{ms}}$",
        )
        axes[0, 1].plot(
            f_arr,
            phase_diff,
            ls=line_styles[i % len(line_styles)],
            color=line_colors[i % len(line_colors)],
            label=rf"$\Delta t_{{\mathrm{{d}}}} = {Delta_td*1000:.2g}\,\mathrm{{ms}}$",
        )

    # --- Bottom row: vary I (via y) at fixed time delay def_td ---
    for i, I in enumerate(np.atleast_1d(I_arr)):
        lens_params = _build_lensing_params_from_I(
            lens_params_base,
            I,
            def_td,
            z,
        )
        lens_inst = LensingGeo(lens_params)
        f_cut = f_cut_common if share_frequency_range else lens_inst.f_cut()
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
            label=rf"$I = {I_val:.2g}$",
        )
        axes[1, 1].plot(
            f_arr,
            phase_diff,
            ls=line_styles[i % len(line_styles)],
            color=line_colors[i % len(line_colors)],
            label=rf"$I = {I_val:.2g}$",
        )

    # Labels and styling
    if amplitude_mode == "abs":
        amp_label = r"$\left|\tilde{\mathit{h}}\right|$"
        customize_2x2_axes(axes)
    else:
        amp_label = (
            r"$\left(\mathit{B}_{\mathrm{L}}/\mathit{B}_{\mathrm{NP}}\right) - 1$"
        )
        customize_2x2_axes_ratio(axes)

    phase_label = r"$\Phi_{\mathrm{L}} - \Phi_{\mathrm{NP}}\,[\mathrm{rad}]$"
    _add_vertical_axis_labels(
        fig,
        axes,
        amp_label,
        phase_label,
        reference_row=0,
        phase_label_x_shift=-0.007,
    )

    if share_frequency_range and f_cut_common is not None:
        for ax in axes.ravel():
            ax.set_xlim(f_min, f_cut_common)

    if z is not None:
        print(f"Applied redshift z={z} (mcz treated as source-frame)")

    if save_path:
        out_path = _build_save_path_with_mcz(save_path, z, lens_params_base["mcz"])
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
    share_frequency_range: bool = True,
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
    share_frequency_range: if True, all panels use a common f range [f_min, f_cut_common]
        for cleaner paper-ready comparison across subplots.
    save_path: optional file path to save the figure; format inferred by extension.
    Returns (fig, axes).
    """
    apply_physics_paper_style(base_font=12, label_font=14, tick_font=11, legend_font=11)
    _validate_amplitude_mode(amplitude_mode)
    if delta_f <= 0:
        raise ValueError("delta_f must be > 0")
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 18), sharex=True)
    fig.subplots_adjust(wspace=0.25, hspace=0.12)

    f_cut_common = None
    if share_frequency_range:
        f_cut_candidates = []

        def _collect_fcuts(vary_key, values, fixed_items):
            for val in np.atleast_1d(values):
                rp_inst, np_inst = _build_precessing_instances(
                    RP_params_1,
                    NP_params_1,
                    vary_key,
                    val,
                    fixed_items,
                    z,
                )
                f_cut_candidates.append(float(min(rp_inst.f_cut(), np_inst.f_cut())))

        _collect_fcuts(
            "omega_tilde",
            omega_vals,
            {"theta_tilde": fixed_theta},
        )
        _collect_fcuts(
            "theta_tilde",
            theta_vals,
            {"omega_tilde": fixed_omega},
        )
        _collect_fcuts(
            "gamma_P",
            gamma_vals,
            {"theta_tilde": fixed_theta, "omega_tilde": fixed_omega},
        )

        f_cut_common = float(np.min(f_cut_candidates))
        if f_cut_common <= f_min:
            raise ValueError("Common f_cut must be greater than f_min")

    # Helper to run a sweep
    def sweep(row, vary_key, values, fixed_items, label_tex):
        for i, val in enumerate(values):
            RP_inst, NP_inst = _build_precessing_instances(
                RP_params_1,
                NP_params_1,
                vary_key,
                val,
                fixed_items,
                z,
            )
            f_cut = (
                f_cut_common
                if share_frequency_range
                else min(RP_inst.f_cut(), NP_inst.f_cut())
            )
            f_arr = make_frequency_array(f_min, f_cut, delta_f=delta_f)
            RP_strain = RP_inst.strain(f_arr)
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
                label_str = r"$\gamma_{\mathrm{P}}$ = " + gamma_tex
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
        r"$\gamma_{\mathrm{P}}$",
    )

    # Labels and styling
    if amplitude_mode == "abs":
        amp_label = r"$\left|\tilde{\mathit{h}}\right|$"
        customize_3x2_axes_abs(axes)
    else:
        amp_label = (
            r"$\left(\mathit{B}_{\mathrm{RP}}/\mathit{B}_{\mathrm{NP}}\right) - 1$"
        )
        customize_3x2_axes_ratio(axes)

    phase_label = r"$\Phi_{\mathrm{RP}} - \Phi_{\mathrm{NP}}\,[\mathrm{rad}]$"
    _add_vertical_axis_labels(
        fig,
        axes,
        amp_label,
        phase_label,
        reference_row=1,
        phase_label_x_shift=0.015,
    )

    if share_frequency_range and f_cut_common is not None:
        for ax in axes.ravel():
            ax.set_xlim(f_min, f_cut_common)

    if z is not None:
        print(f"Applied redshift z={z} (mcz treated as source-frame)")

    if save_path:
        out_path = _build_save_path_with_mcz(save_path, z, RP_params_1["mcz"])
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
    apply_physics_paper_style(base_font=12, label_font=14, tick_font=11, legend_font=11)
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
