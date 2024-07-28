import numpy as np

error_handler = np.seterr(invalid="raise")
from pycbc.types import FrequencySeries
from modules.default_params_ver2 import *

NEAR_ZERO_THRESHOLD = 1e-8

# ONLY APPLICABLE FOR ARRAYS OF PARAMETERS, NOT SINGLE VALUES


def L_total_mass(mcz=20 * solar_mass, eta=0.25, **kwargs):
    """Total mass from chirp mass [seconds]"""
    return mcz / (eta ** (3 / 5))


def L_f_cut(**kwargs):
    """L_f_cut"""
    return 1 / (6 ** (3 / 2) * np.pi * L_total_mass(**kwargs))


def L_LdotN(
    theta_S=np.pi / 4, phi_S=0.0, theta_J=8 * np.pi / 9, phi_J=np.pi / 4, **kwargs
):
    """(cosine angle between l and n)"""
    cos_term = np.cos(theta_S) * np.cos(theta_J)
    sin_term = np.sin(theta_S) * np.sin(theta_J) * np.cos(phi_S - phi_J)
    inner_prod = cos_term + sin_term
    return inner_prod


def L_amp(dist=1.5 * giga_parsec, mcz=20 * solar_mass, **kwargs):
    """A for h(f)"""
    amplitude = np.sqrt(5 / 96) * np.pi ** (-2 / 3) * mcz ** (5 / 6) / (dist)
    return amplitude


def L_Psi(f, mcz=20 * solar_mass, eta=0.25, t_c=0.0, phi_c=0.0, **kwargs):
    """eqn 3.13 in Cutler-Flanaghan 1994"""
    x = (np.pi * L_total_mass(**kwargs) * f) ** (2 / 3)
    term1 = 2 * np.pi * f * t_c - phi_c - np.pi / 4
    prefactor = (3 / 4) * (8 * np.pi * mcz * f) ** (-5 / 3)
    term2 = 1 + (20 / 9) * (743 / 336 + (11 / 4) * eta) * x - 16 * np.pi * x ** (3 / 2)
    L_Psi = term1 + prefactor * term2
    return L_Psi


def L_psi_s(
    theta_S=np.pi / 4, phi_S=0.0, theta_J=8 * np.pi / 9, phi_J=np.pi / 4, **kwargs
):
    """L_psi_s that goes into F_plus and F_cross"""

    numerator = np.cos(theta_J) - np.cos(theta_S) * (L_LdotN(**kwargs))
    denominator = np.sin(theta_S) * np.sin(theta_J) * np.sin(phi_J - phi_S)

    psi_s_val = np.arctan2(numerator, denominator)
    return psi_s_val


def L_fIp(theta_S=np.pi / 4, phi_S=0.0, **kwargs):
    """F_plus"""

    term_1 = (
        1
        / 2
        * (1 + np.power(np.cos(theta_S), 2))
        * np.cos(2 * phi_S)
        * np.cos(2 * L_psi_s(**kwargs))
    )
    term_2 = np.cos(theta_S) * np.sin(2 * phi_S) * np.sin(2 * L_psi_s(**kwargs))

    fIp_val = term_1 - term_2
    return fIp_val


def L_fIc(theta_S=np.pi / 4, phi_S=0.0, **kwargs):
    """F_cross"""

    term_1 = (
        1
        / 2
        * (1 + np.power(np.cos(theta_S), 2))
        * np.cos(2 * phi_S)
        * np.sin(2 * L_psi_s(**kwargs))
    )
    term_2 = np.cos(theta_S) * np.sin(2 * phi_S) * np.cos(2 * L_psi_s(**kwargs))

    fIc_val = term_1 + term_2
    return fIc_val


def L_lambdaI(**kwargs):
    """|F_plus (1+L.N**2) - i (2*F_cross*L.N)|"""

    term_1 = np.power(2 * L_LdotN(**kwargs) * L_fIc(**kwargs), 2)
    term_2 = np.power((1 + np.power(L_LdotN(**kwargs), 2)) * L_fIp(**kwargs), 2)
    lambdaI_val = np.sqrt(term_1 + term_2)
    return lambdaI_val


def L_phi_pI(**kwargs):
    """tan-1((2*F_cross*L.N)/F_plus (1+L.N**2))"""

    numerator = 2 * L_LdotN(**kwargs) * L_fIc(**kwargs)
    denominator = (1 + np.power(L_LdotN(**kwargs), 2)) * L_fIp(**kwargs)

    phi_pI_val = np.arctan2(numerator, denominator)
    return phi_pI_val


def L_hI(f, **kwargs):
    """Unlensed Waveform"""

    term_1 = L_lambdaI(**kwargs)
    term_2 = np.exp(-1j * L_phi_pI(**kwargs))
    term_3 = L_amp(**kwargs) * f ** (-7 / 6)
    term_4 = np.exp(1j * L_Psi(f, **kwargs))

    signal_I = term_1 * term_2 * term_3 * term_4

    return signal_I


def L_mu_plus(y=0.25, **kwargs):
    """plus magnification, equation 18 in Takahashi & Nakamura 2003, also 16a in Saif et al. 2023"""
    mu_plus_val = 1 / 2 + (y**2 + 2) / (2 * y * np.sqrt(y**2 + 4)) + 0j
    return mu_plus_val


def L_mu_minus(y=0.25, **kwargs):
    """minus magnification, equation 18 in Takahashi & Nakamura 2003, also 16a in Saif et al. 2023"""
    mu_minus_val = 1 / 2 - (y**2 + 2) / (2 * y * np.sqrt(y**2 + 4)) + 0j
    return mu_minus_val


def L_I(**kwargs):
    """flux ratio, equation 17a in Saif et al. 2023"""
    I_val = np.abs(L_mu_minus(**kwargs)) / np.abs(L_mu_plus(**kwargs))
    return I_val


def L_td(MLz=2e3 * solar_mass, y=0.25, **kwargs):
    """time delay, equation 16b in Saif et al. 2023"""
    td_val = (
        2
        * MLz
        * (
            y * np.sqrt(y**2 + 4)
            + 2 * np.log((np.sqrt(y**2 + 4) + y) / (np.sqrt(y**2 + 4) - y))
        )
    )
    return td_val


def L_F(f, **kwargs):
    """PM amplification factor in geometric optics limit, equation 18 in Takahashi & Nakamura 2003"""
    L_mu_plus_arr = L_mu_plus(**kwargs)
    L_mu_plus_arr = L_mu_plus_arr[:, np.newaxis]  # Make it 2D for broadcasting
    L_mu_minus_arr = L_mu_minus(**kwargs)
    L_mu_minus_arr = L_mu_minus_arr[:, np.newaxis]  # Make it 2D for broadcasting
    L_td_arr = L_td(**kwargs)
    L_td_arr = L_td_arr[:, np.newaxis]  # Make it 2D for broadcasting
    f = np.tile(f, L_td_arr.shape)

    F_val = np.sqrt(np.abs(L_mu_plus_arr)) - 1j * np.sqrt(
        np.abs(L_mu_minus_arr)
    ) * np.exp(2j * np.pi * f * L_td_arr)
    return F_val


def L_strain(f, delta_f=0.25, frequencySeries=True, **kwargs):
    """lensed strain = unlensed strain * amplification factor
    Args:
        f (numpy array): frequency range
        delta_f (float): interval length of frequency. Default at 0.25 Hz.
        frequencySeries (bool): True for FrequencySeries. False otherwise.

    Returns:
        hL (numpy array): lensed strain
    """
    hL = L_hI(f, **kwargs) * L_F(f, **kwargs)

    if frequencySeries:
        hL = list(hL)
        for i in range(len(hL)):
            hL[i] = FrequencySeries(hL[i], delta_f)
        return hL

    return hL
