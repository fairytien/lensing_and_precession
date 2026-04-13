"""Waveform and parameter utility functions.

This module is a source-of-truth implementation for core helpers formerly
kept in `functions_v3.py`.
"""

import copy
from typing import Union

import numpy as np
from numpy.lib import NumpyVersion
from scipy.optimize import fsolve

from modules.Classes import LensingGeo, Precessing
from modules.default_params import SOLMASS2SEC


# Compatibility shim for NumPy 1.24+ where several aliases were removed.
if not hasattr(np, "asscalar"):
    np.asscalar = lambda a: a.item()
if not hasattr(np, "alen"):
    np.alen = lambda a: len(a)
if NumpyVersion(np.__version__) < NumpyVersion("1.24.0"):
    for _name, _alias in (
        ("float", float),
        ("int", int),
        ("bool", bool),
        ("complex", complex),
        ("object", object),
    ):
        if not hasattr(np, _name):
            setattr(np, _name, _alias)


def set_to_params(*args):
    """Return deep-copied parameter dictionaries."""
    args_copy = [copy.deepcopy(arg) for arg in args]
    return tuple(args_copy)


def set_orientation(orient_dict: dict, *args):
    """Apply orientation angles to one or more parameter dictionaries."""
    args_copy = [copy.deepcopy(arg) for arg in args]

    for arg_copy in args_copy:
        arg_copy["theta_J"] = orient_dict["theta_J"]
        arg_copy["phi_J"] = orient_dict["phi_J"]
        arg_copy["theta_S"] = orient_dict["theta_S"]
        arg_copy["phi_S"] = orient_dict["phi_S"]

    return tuple(args_copy)


def get_gw(
    params: dict,
    f_min: float = 20,
    delta_f: float = 0.25,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    frequencySeries=True,
) -> dict:
    """Compute strain/phase/frequency arrays for lensing or precessing params."""
    if "MLz" in params and "y" in params:
        gw_inst = lens_Class(params)
    else:
        gw_inst = prec_Class(params)

    f_cut = gw_inst.f_cut()
    f_arr = np.arange(f_min, f_cut, delta_f)
    strain = gw_inst.strain(f_arr, delta_f, frequencySeries)
    phase = np.unwrap(np.angle(strain))

    return {"strain": strain, "phase": phase, "f_array": f_arr}


def get_MLz_from_td(td, y) -> Union[float, np.ndarray]:
    """Return lens mass [Msun] from time delay [s] and source position y."""
    divisor = 2 * (
        y * np.sqrt(y**2 + 4)
        + 2 * np.log((np.sqrt(y**2 + 4) + y) / (np.sqrt(y**2 + 4) - y))
    )
    return (td / divisor) / SOLMASS2SEC


def get_td_from_MLz(MLz, y) -> Union[float, np.ndarray]:
    """Return time delay [s] from lens mass [Msun] and source position y."""
    td_val = (
        2
        * MLz
        * SOLMASS2SEC
        * (
            y * np.sqrt(y**2 + 4)
            + 2 * np.log((np.sqrt(y**2 + 4) + y) / (np.sqrt(y**2 + 4) - y))
        )
    )
    return td_val


def get_I_from_y(y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Return flux ratio I from source position y."""
    mu_plus = 1 / 2 + (y**2 + 2) / (2 * y * np.sqrt(y**2 + 4)) + 0j
    mu_minus = 1 / 2 - (y**2 + 2) / (2 * y * np.sqrt(y**2 + 4)) + 0j
    return np.abs(mu_minus) / np.abs(mu_plus)


def get_y_from_I(I: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Return source position y from flux ratio I (0 < I < 1)."""
    scalar_input = np.isscalar(I) or (np.ndim(I) == 0)
    I_arr = np.atleast_1d(np.asarray(I, dtype=float))

    if np.any(I_arr >= 1):
        raise ValueError("Flux ratio must be less than 1.")
    if np.any(I_arr <= 0):
        raise ValueError("Flux ratio must be positive.")

    y_roots = np.empty_like(I_arr, dtype=float)
    for idx, I_val in np.ndenumerate(I_arr):
        y_roots[idx] = fsolve(lambda y: get_I_from_y(y) - I_val, 1.0)[0]

    if scalar_input:
        return float(y_roots.ravel()[0])
    return y_roots


def get_fcut_from_mcz(mcz, eta=0.25) -> Union[float, np.ndarray]:
    """Return ISCO cutoff frequency [Hz] from chirp mass [Msun]."""
    return eta ** (3 / 5) / (6 ** (3 / 2) * np.pi * mcz * SOLMASS2SEC)


def get_mcz_from_fcut(fcut, eta=0.25) -> Union[float, np.ndarray]:
    """Return chirp mass [Msun] from ISCO cutoff frequency [Hz]."""
    return eta ** (3 / 5) / (6 ** (3 / 2) * np.pi * fcut) / SOLMASS2SEC


def number_of_prec_cycles(
    mcz_msun: Union[float, np.ndarray],
    omega_tilde: Union[float, np.ndarray],
    f_min: float = 20,
    eta: float = 0.25,
) -> Union[float, np.ndarray]:
    """Compute precession-cycle count between f_min and f_cut."""
    mcz_msun = np.asarray(mcz_msun)
    omega_tilde = np.asarray(omega_tilde)
    f_cut = get_fcut_from_mcz(mcz_msun, eta)
    mcz_sec = mcz_msun * SOLMASS2SEC
    M_sec = mcz_sec / (eta ** (3.0 / 5.0))
    denom = (
        (M_sec / SOLMASS2SEC)
        * (np.pi ** (8.0 / 3.0))
        * (mcz_sec ** (5.0 / 3.0))
        * (f_cut ** (5.0 / 3.0))
    )
    A = (5000.0 / 96.0) / denom
    phi_LJ_cut = (A * omega_tilde) * (1.0 / f_min - 1.0 / f_cut)
    return phi_LJ_cut / (2 * np.pi)


def number_of_lens_cycles(
    mcz_msun: Union[float, np.ndarray],
    td: Union[float, np.ndarray],
    f_min: float = 20,
    eta: float = 0.25,
) -> Union[float, np.ndarray]:
    """Compute lensing modulation-cycle count between f_min and f_cut."""
    mcz_msun = np.asarray(mcz_msun)
    td = np.asarray(td)
    f_cut = get_fcut_from_mcz(mcz_msun, eta)
    return (f_cut - f_min) * td


def mcz_for_n_lens_cycles(
    n_cycles: float, td: Union[float, np.ndarray], f_min: float = 20, eta: float = 0.25
) -> Union[float, np.ndarray]:
    """Invert lens-cycle count to chirp mass [Msun]."""
    f_cut = f_min + (n_cycles / td)
    return get_mcz_from_fcut(f_cut, eta)


def mcz_for_n_prec_cycles(
    n_cycles: float,
    omega_tilde: Union[float, np.ndarray],
    f_min: float = 20,
    eta: float = 0.25,
) -> Union[float, np.ndarray]:
    """Invert precession-cycle count to chirp mass [Msun]."""
    n_cycles = np.asarray(n_cycles, dtype=float)
    omega_tilde = np.asarray(omega_tilde, dtype=float)

    K = (6 ** (3.0 / 2.0)) * np.pi * SOLMASS2SEC / (eta ** (3.0 / 5.0))
    C = ((5000.0 / 96.0) * (6 ** (5.0 / 2.0))) / (
        2.0 * (np.pi**2) * (eta ** (2.0 / 5.0))
    )

    if np.any(omega_tilde == 0):
        raise ValueError("omega_tilde must be non-zero to invert n_prec -> mcz")

    denom = f_min * (n_cycles / (C * omega_tilde) + K)
    if np.any(denom <= 0):
        raise ValueError(
            "Requested n_cycles and omega_tilde lead to non-physical mcz (denominator <= 0)"
        )

    return 1.0 / denom


def get_lens_limits_for_RP_L(
    mcz_msun: float,
    omega_tilde: float,
    lower: Union[str, float] = "min",
    upper: Union[str, float] = "max",
    y: float = 0.25,
    f_min: float = 20,
    eta: float = 0.25,
) -> dict:
    """Compute MLz and td bounds for lensing-vs-precession cycle comparison."""
    f_cut = get_fcut_from_mcz(mcz_msun, eta)

    if lower == "min":
        MLz_min = (1 / (8 * np.pi * f_min)) / SOLMASS2SEC
        td_min = get_td_from_MLz(MLz_min, y)
    elif isinstance(lower, float):
        td_min = lower / (f_cut - f_min)
        MLz_min = get_MLz_from_td(td_min, y)
    else:
        raise ValueError("lower must be 'min' or a float.")

    if upper == "max":
        n_prec_cycles = number_of_prec_cycles(
            mcz_msun, omega_tilde, f_min=f_min, eta=eta
        )
        td_max = n_prec_cycles / (f_cut - f_min)
        MLz_max = get_MLz_from_td(td_max, y)
    elif isinstance(upper, float):
        td_max = upper / (f_cut - f_min)
        MLz_max = get_MLz_from_td(td_max, y)
    else:
        raise ValueError("upper must be 'max' or a float.")

    return {
        "MLz_min": MLz_min,
        "MLz_max": MLz_max,
        "td_min": td_min,
        "td_max": td_max,
    }
