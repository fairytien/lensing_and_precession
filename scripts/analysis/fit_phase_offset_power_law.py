"""Fit a power-law phase-offset model from extracted best-fit precession data.

The fitted model is

    value = C * theta_tilde**a * omega_tilde**b

which becomes linear in log space:

    log(value) = log(C) + a * log(theta_tilde) + b * log(omega_tilde)

Input data can be a headered text/CSV file with named columns or a plain
three-column file ordered as

    theta_tilde, omega_tilde, value

Example:

    python -m scripts.analysis.fit_phase_offset_power_law \
        --input data/system1_phase_offsets.csv \
        --title "System 1 (Taman_faceon)"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure repository root is importable when running this file directly.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.plot_utils import apply_physics_paper_style, save_figure

apply_physics_paper_style()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit value = C * theta_tilde^a * omega_tilde^b in log space from "
            "an extracted data table."
        )
    )
    parser.add_argument("--input", required=True, help="Path to the input table.")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Prefix for output files. Defaults to figures/analysis/<input_stem>_power_law"
        ),
    )
    parser.add_argument(
        "--title",
        default="Power-law fit",
        help="Figure title prefix.",
    )
    parser.add_argument(
        "--value-label",
        default=r"$\Delta\Phi_{\mathrm{P}}\,[\mathrm{rad}]$",
        help="Axis label for the fitted dependent variable.",
    )
    return parser.parse_args()


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delimiter = "," if path.suffix.lower() == ".csv" else None
    table = np.genfromtxt(
        path, delimiter=delimiter, names=True, dtype=float, encoding=None
    )
    if table.dtype.names is None:
        raise ValueError("Expected a header row in the input file.")
    cols = ("theta_tilde", "omega_tilde", "delta_phi")
    missing = [c for c in cols if c not in table.dtype.names]
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}. "
            f"Available: {', '.join(table.dtype.names)}"
        )
    return (
        np.asarray(table["theta_tilde"], dtype=float),
        np.asarray(table["omega_tilde"], dtype=float),
        np.asarray(table["delta_phi"], dtype=float),
    )


def validate_data(
    theta: np.ndarray, omega: np.ndarray, value: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite_mask = np.isfinite(theta) & np.isfinite(omega) & np.isfinite(value)
    theta = theta[finite_mask]
    omega = omega[finite_mask]
    value = value[finite_mask]

    if theta.size < 3:
        raise ValueError("Need at least three finite rows to fit the power law.")

    if np.any(theta <= 0) or np.any(omega <= 0) or np.any(value <= 0):
        raise ValueError(
            "All theta_tilde, omega_tilde, and value entries must be strictly positive "
            "for a log-space fit."
        )

    return theta, omega, value


def fit_power_law(
    theta: np.ndarray, omega: np.ndarray, value: np.ndarray
) -> tuple[float, float, float, float]:
    log_theta = np.log(theta)
    log_omega = np.log(omega)
    log_value = np.log(value)

    design = np.column_stack([np.ones(theta.size), log_theta, log_omega])
    coeffs, _, _, _ = np.linalg.lstsq(design, log_value, rcond=None)

    log_c, exponent_theta, exponent_omega = coeffs
    fitted_log_value = design @ coeffs
    ss_res = np.sum((log_value - fitted_log_value) ** 2)
    ss_tot = np.sum((log_value - np.mean(log_value)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return (
        float(np.exp(log_c)),
        float(exponent_theta),
        float(exponent_omega),
        float(r_squared),
    )


def nice_log_bounds(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        raise ValueError("Cannot compute log bounds for an empty array.")

    lower_value = float(np.min(values))
    upper_value = float(np.max(values))
    lower_exp = int(np.floor(np.log10(lower_value)))
    upper_exp = int(np.floor(np.log10(upper_value)))

    mantissas = np.array([1.0, 2.0, 5.0, 10.0])
    lower_scale = 10.0**lower_exp
    upper_scale = 10.0**upper_exp

    lower_candidates = mantissas * lower_scale
    upper_candidates = mantissas * upper_scale

    lower = float(lower_candidates[lower_candidates <= lower_value][-1])
    upper = float(upper_candidates[upper_candidates >= upper_value][0])

    if np.isclose(lower, upper):
        upper *= 10.0
    return lower, upper


def nice_linear_bounds(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot compute linear bounds for an empty array.")

    lower_value = float(np.min(values))
    upper_value = float(np.max(values))
    span = upper_value - lower_value
    if span <= 0:
        pad = 0.5 * max(abs(lower_value), 1.0)
        return lower_value - pad, upper_value + pad

    scale = 10.0 ** np.floor(np.log10(span))
    for factor in (0.1, 0.2, 0.5, 1.0):
        step = factor * scale
        rounded_lower = step * np.floor(lower_value / step)
        rounded_upper = step * np.ceil(upper_value / step)
        if rounded_lower < rounded_upper:
            return float(rounded_lower), float(rounded_upper)

    return lower_value, upper_value


def make_loglog_plot(
    theta: np.ndarray,
    omega: np.ndarray,
    value: np.ndarray,
    *,
    constant: float,
    exponent_theta: float,
    exponent_omega: float,
    value_label: str,
    title: str,
    output_path: Path,
) -> None:
    predictor = theta**exponent_theta * omega**exponent_omega
    order = np.argsort(predictor)
    predictor_sorted = predictor[order]
    value_sorted = value[order]

    x_fit = np.geomspace(np.min(predictor_sorted), np.max(predictor_sorted), 300)
    y_fit = constant * x_fit

    fig, ax = plt.subplots(figsize=(6.9, 5.4))
    ax.scatter(predictor_sorted, value_sorted, s=34, color="C0", label="Data", zorder=3)
    ax.plot(x_fit, y_fit, color="C1", linewidth=2.2, label="Best fit")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(
        rf"$\tilde{{\theta}}^{{{exponent_theta:.2f}}}\,\tilde{{\Omega}}^{{{exponent_omega:.2f}}}$"
    )
    ax.set_ylabel(value_label)
    ax.set_xlim(*nice_log_bounds(predictor_sorted))
    ax.set_ylim(*nice_log_bounds(value_sorted))
    ax.grid(True, which="both", alpha=0.25, linewidth=0.8)
    ax.legend(loc="best", frameon=True)
    ax.set_title(f"{title}: log-log collapse")
    ax.margins(x=0.0, y=0.0)

    model_text = (
        rf"$C={constant:.3g}$"
        + "\n"
        + rf"$a={exponent_theta:.3f}$"
        + "\n"
        + rf"$b={exponent_omega:.3f}$"
    )
    ax.text(
        0.03,
        0.97,
        model_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "0.7",
        },
    )

    save_figure(fig, output_path)


def make_surface_plot(
    theta: np.ndarray,
    omega: np.ndarray,
    *,
    constant: float,
    exponent_theta: float,
    exponent_omega: float,
    value_label: str,
    title: str,
    output_path: Path,
) -> None:
    omega_grid = np.linspace(np.min(omega), np.max(omega), 250)
    theta_grid = np.linspace(np.min(theta), np.max(theta), 250)
    omega_mesh, theta_mesh = np.meshgrid(omega_grid, theta_grid)
    fitted_value = constant * theta_mesh**exponent_theta * omega_mesh**exponent_omega

    fig, ax = plt.subplots(figsize=(7.3, 5.9))
    contour = ax.contourf(omega_mesh, theta_mesh, fitted_value, levels=40, cmap="jet")
    ax.set_xlabel(r"$\tilde{\Omega}$")
    ax.set_ylabel(r"$\tilde{\theta}$")
    ax.set_xlim(*nice_linear_bounds(omega))
    ax.set_ylim(*nice_linear_bounds(theta))
    ax.set_title(f"{title}: fitted power-law surface")
    ax.margins(x=0.0, y=0.0)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(value_label)

    save_figure(fig, output_path)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    theta, omega, value = load_data(input_path)
    theta, omega, value = validate_data(theta, omega, value)

    constant, exponent_theta, exponent_omega, r_squared = fit_power_law(
        theta, omega, value
    )

    output_prefix = (
        Path(args.output_prefix)
        if args.output_prefix is not None
        else Path("figures") / "analysis" / f"{input_path.stem}_power_law"
    )
    loglog_path = output_prefix.with_name(output_prefix.name + "_loglog.png")
    surface_path = output_prefix.with_name(output_prefix.name + "_surface.png")

    make_loglog_plot(
        theta,
        omega,
        value,
        constant=constant,
        exponent_theta=exponent_theta,
        exponent_omega=exponent_omega,
        value_label=args.value_label,
        title=args.title,
        output_path=loglog_path,
    )
    make_surface_plot(
        theta,
        omega,
        constant=constant,
        exponent_theta=exponent_theta,
        exponent_omega=exponent_omega,
        value_label=args.value_label,
        title=args.title,
        output_path=surface_path,
    )

    print("Fitted model: value = C * theta_tilde^a * omega_tilde^b")
    print(f"C = {constant:.8g}")
    print(f"a = {exponent_theta:.8g}")
    print(f"b = {exponent_omega:.8g}")
    print(f"R^2 in log space = {r_squared:.6f}")


if __name__ == "__main__":
    main()
