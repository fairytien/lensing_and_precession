"""Noise-curve and SNR utilities.

This module is a source-of-truth implementation for PSD/SNR helpers formerly
kept in `functions_v3.py`.
"""

from typing import Union

import numpy as np
from pycbc.types import FrequencySeries
from scipy.integrate import simpson

from modules.Classes import LensingGeo, Precessing


def Sn(
    f_arr: np.ndarray,
    f_min: float = 20,
    delta_f: float = 0.25,
    frequencySeries=True,
) -> Union[np.ndarray, FrequencySeries]:
    """Return aLIGO design PSD approximation for given frequencies."""
    f_arr = np.asarray(f_arr)
    S0 = 1e-49
    f0 = 215.0
    x = f_arr / f0

    Sn_temp = (
        np.power(x, -4.14)
        - 5.0 * np.power(x, -2.0)
        + 111.0 * ((1.0 - x**2 + 0.5 * x**4) / (1.0 + 0.5 * x**2))
    )
    Sn_val = Sn_temp * S0
    Sn_val = np.where(f_arr < f_min, np.inf, Sn_val)

    if frequencySeries:
        return FrequencySeries(Sn_val, delta_f=delta_f)
    return Sn_val


def SNR(
    params: dict,
    f_min: float = 20,
    delta_f: float = 0.25,
    psd: FrequencySeries | None = None,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
) -> float:
    """Compute matched-filter SNR for a lensing or precessing waveform."""
    if "MLz" in params and "y" in params:
        gw_inst = lens_Class(params)
    else:
        gw_inst = prec_Class(params)

    f_cut = gw_inst.f_cut()
    f_arr = np.arange(f_min, f_cut, delta_f)
    _psd = psd if psd is not None else Sn(f_arr, f_min=f_min, delta_f=delta_f)
    h = gw_inst.strain(f_arr, delta_f=delta_f)

    integrand = np.abs(h) ** 2 / _psd
    integrated_inner_product = simpson(integrand, x=f_arr)
    snr = np.sqrt(4 * np.real(integrated_inner_product))

    return float(snr)
