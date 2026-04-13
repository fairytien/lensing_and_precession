#############################
# Section 1: Import Modules #
#############################


# if running on Google Colab, uncomment the following lines
# import sys
# !{sys.executable} -m pip install pycbc ligo-common --no-cache-dir

# import py scripts
# Production defaults are pinned to Classes_v2 for numerical stability.
from modules.Classes import LensingGeo, Precessing
from modules.default_params import (
    GIGAPC2SEC,
    NP_params_0,
    NP_params_1,
    RP_params_0,
    RP_params_1,
    SOLMASS2SEC,
    YEAR2SEC,
    error_handler,
    lens_params_0,
    lens_params_1,
    loc_params,
    np,
    omega_theta_tilde_pairs,
    orient_params,
)

# import modules
import numpy as np
from numpy.lib import NumpyVersion

# Compatibility shim for NumPy 1.24+ where several aliases were removed
# astropy still expects np.asscalar to exist
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

error_handler = np.seterr(invalid="raise")
from scipy.integrate import simpson
from scipy.optimize import fsolve, minimize_scalar
from pycbc.types import FrequencySeries, TimeSeries
import os
from datetime import datetime
import time
import pickle
import copy
from typing import Union, Type, Tuple, List, Dict, Any, Optional


######################################
# Section 2: Shortcuts & Convenience #
######################################


def set_to_params(*args):
    """
    Returns a tuple of deep copies of the input arguments.

    Args:
        *args: Any number of arguments to be deep copied.

    Returns:
        A tuple of deep copies of the input arguments.

    Example:
        >>> params, _ = set_to_params(RP_params_1)
    """

    args_copy = [copy.deepcopy(arg) for arg in args]
    return tuple(args_copy)


def set_orientation(orient_dict: dict, *args):
    """
    Sets the angular orientation of each argument in `args` to the values specified in `orient_dict`.

    Args:
        orient_dict (dict): A dictionary containing the angular orientation values to set for each argument.
        *args: One or more dictionaries representing the arguments to modify.

    Returns:
        tuple: A tuple containing the modified versions of each argument in `args`.

    Example:
        >>> params, _ = set_orientation(orient_params["Taman"]["edgeon"], params)
    """

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
    """
    Calculates the GW for a given set of parameters.

    Args:
        params (dict): A dictionary containing the parameters.
        f_min (float): The minimum frequency.
        delta_f (float): The frequency step size.
        lens_Class (class): The class for lensing parameters.
        prec_Class (class): The class for precessing parameters.

    Returns:
        dict: A dictionary containing the following keys:
        - "strain" (np.ndarray or FrequencySeries): The complex strain.
        - "phase" (np.ndarray): The GW phase.
        - "f_array" (np.ndarray): The frequency array.
    """
    if "MLz" in params and "y" in params:  # lensing parameters, use lens_Class
        gw_inst = lens_Class(params)
    else:  # precessing parameters, use prec_Class
        gw_inst = prec_Class(params)

    f_cut = gw_inst.f_cut()
    f_arr = np.arange(f_min, f_cut, delta_f)
    strain = gw_inst.strain(f_arr, delta_f, frequencySeries)
    phase = np.unwrap(np.angle(strain))

    return {"strain": strain, "phase": phase, "f_array": f_arr}


def get_MLz_from_td(td, y) -> Union[float, np.ndarray]:
    """
    Calculates the lens mass [solar mass] from the given time delay [second] and source position [dimensionless].

    Args:
        td (float or ndarray): The time delay [second].
        y (float or ndarray): The source position of the lens [dimensionless].

    Returns:
        float or ndarray: The calculated lens mass [solar mass].
    """
    divisor = 2 * (
        y * np.sqrt(y**2 + 4)
        + 2 * np.log((np.sqrt(y**2 + 4) + y) / (np.sqrt(y**2 + 4) - y))
    )
    return (td / divisor) / SOLMASS2SEC


def get_td_from_MLz(MLz, y) -> Union[float, np.ndarray]:
    """
    Calculates the time delay [second] from the given lens mass [solar mass] and source position [dimensionless], based on equation 16b in Saif et al. 2023.

    Args:
        MLz (float or ndarray): The lens mass [solar mass].
        y (float or ndarray): The source position of the lens [dimensionless].

    Returns:
        float or ndarray: The calculated time delay [second].
    """
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
    """
    Calculates the flux ratio [dimensionless] from the given source position [dimensionless], based on equations 16-17 in Saif et al. 2023.

    Args:
        y (float or ndarray): The source position of the lens [dimensionless].

    Returns:
        float or ndarray: The calculated flux ratio [dimensionless].
    """
    # plus magnification, equation 18 in Takahashi & Nakamura 2003, also 16a in Saif et al. 2023
    mu_plus = 1 / 2 + (y**2 + 2) / (2 * y * np.sqrt(y**2 + 4)) + 0j

    # minus magnification, equation 18 in Takahashi & Nakamura 2003, also 16a in Saif et al. 2023
    mu_minus = 1 / 2 - (y**2 + 2) / (2 * y * np.sqrt(y**2 + 4)) + 0j

    return np.abs(mu_minus) / np.abs(mu_plus)


def get_y_from_I(I: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculates the source position [dimensionless] from the given flux ratio [dimensionless]. Assumes 0 < I < 1 for valid calculations (positive y).

    Args:
        I (float or ndarray): The flux ratio [dimensionless]. Must be 0 < I < 1.

    Returns:
        float or ndarray: The calculated source position [dimensionless]. For ndarray inputs, returns an ndarray of source positions corresponding to each flux ratio.
    """
    # Normalize input to at least 1D array, track whether input was scalar
    scalar_input = np.isscalar(I) or (np.ndim(I) == 0)
    I_arr = np.atleast_1d(np.asarray(I, dtype=float))

    # Validate domain
    if np.any(I_arr >= 1):
        raise ValueError("Flux ratio must be less than 1.")
    if np.any(I_arr <= 0):
        raise ValueError("Flux ratio must be positive.")

    # Solve for y for each element (fsolve is not vectorized)
    y_roots = np.empty_like(I_arr, dtype=float)
    for idx, I_val in np.ndenumerate(I_arr):
        y_roots[idx] = fsolve(lambda y: get_I_from_y(y) - I_val, 1.0)[0]

    if scalar_input:
        return float(y_roots.ravel()[0])
    return y_roots


def get_fcut_from_mcz(mcz, eta=0.25) -> Union[float, np.ndarray]:
    """
    Calculates f_cut [Hz] from the given mcz [solar mass] and eta [dimensionless].

    Args:
        mcz (float or np.ndarray): The chirp mass [solar mass].
        eta (float): The symmetric mass ratio [dimensionless]. Default is 0.25.

    Returns:
        float or np.ndarray: The calculated cutoff frequency at ISCO [Hz].
    """
    return eta ** (3 / 5) / (6 ** (3 / 2) * np.pi * mcz * SOLMASS2SEC)


def get_mcz_from_fcut(fcut, eta=0.25) -> Union[float, np.ndarray]:
    """
    Calculates mcz [solar mass] from the given f_cut [Hz] and eta [dimensionless].

    Args:
        fcut (float or np.ndarray): The cutoff frequency at ISCO [Hz].
        eta (float): The symmetric mass ratio [dimensionless]. Default is 0.25.

    Returns:
        float or np.ndarray: The calculated chirp mass [solar mass].
    """
    return eta ** (3 / 5) / (6 ** (3 / 2) * np.pi * fcut) / SOLMASS2SEC


def number_of_prec_cycles(
    mcz_msun: Union[float, np.ndarray],
    omega_tilde: Union[float, np.ndarray],
    f_min: float = 20,
    eta: float = 0.25,
) -> Union[float, np.ndarray]:
    """
    Vectorized calculation of the number of precession cycles between the minimum frequency and the cutoff frequency.

    Args:
        mcz_msun (float or np.ndarray): Chirp mass [solar masses].
        omega_tilde (float or np.ndarray): Precession amplitude parameter.
        f_min (float, optional): The minimum frequency. Default is 20 Hz.
        eta (float, optional): Symmetric mass ratio. Default is 0.25.

    Returns:
        n_cycles (float or np.ndarray): The number of precession cycles between the minimum frequency and the cutoff frequency.
    """
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
    # phi_LJ_min is always zero since (1/f_min - 1/f_min) = 0
    phi_LJ_cut = (A * omega_tilde) * (1.0 / f_min - 1.0 / f_cut)
    n_cycles = phi_LJ_cut / (2 * np.pi)
    return n_cycles


def number_of_lens_cycles(
    mcz_msun: Union[float, np.ndarray],
    td: Union[float, np.ndarray],
    f_min: float = 20,
    eta: float = 0.25,
) -> Union[float, np.ndarray]:
    """
    Vectorized calculation of the number of modulation cycles in the lensed waveform between the minimum frequency and the cutoff frequency.

    Args:
        mcz_msun (float or np.ndarray): Chirp mass in solar masses.
        td (float or np.ndarray): Time delay in seconds.
        f_min (float, optional): The minimum frequency. Default is 20 Hz.
        eta (float, optional): Symmetric mass ratio. Default is 0.25.

    Returns:
        n_cycles (float or np.ndarray): The number of modulation cycles in the lensed waveform between the minimum frequency and the cutoff frequency.
    """
    mcz_msun = np.asarray(mcz_msun)
    td = np.asarray(td)
    f_cut = get_fcut_from_mcz(mcz_msun, eta)
    n_cycles = (f_cut - f_min) * td
    return n_cycles


def mcz_for_n_lens_cycles(
    n_cycles: float, td: Union[float, np.ndarray], f_min: float = 20, eta: float = 0.25
) -> Union[float, np.ndarray]:
    # f_cut = f_min + n/td => mcz = eta^(3/5) / [6^(3/2) * pi * solar_mass * (f_min + n/td)]
    f_cut = f_min + (n_cycles / td)
    return get_mcz_from_fcut(f_cut, eta)


def mcz_for_n_prec_cycles(
    n_cycles: float,
    omega_tilde: Union[float, np.ndarray],
    f_min: float = 20,
    eta: float = 0.25,
) -> Union[float, np.ndarray]:
    """
    Invert number_of_prec_cycles to obtain the chirp mass (in solar masses) that
    yields the requested number of precession cycles between f_min and f_cut.

    Uses the analytic simplification of number_of_prec_cycles where the dependence
    on mcz appears only via a linear term in 1/f_cut and a 1/mcz factor, leading to:

        n_prec = C(eta, f_min) * (1/(f_min * mcz) - K(eta)) * omega_tilde

    which can be inverted for mcz as:

        mcz = 1.0 / ( f_min * (n_prec/(C*omega_tilde) + K) )

    where
        K(eta) = 6^(3/2) * pi * SOLMASS2SEC / eta^(3/5)
        C(eta) = (5000/96) * 6^(5/2) / (2 * pi^2 * eta^(2/5))

    Args:
        n_cycles: Desired number of precession cycles (scalar or ndarray)
        omega_tilde: Precession amplitude parameter (scalar or ndarray)
        f_min: Lower frequency cutoff [Hz] (default: 20)
        eta: Symmetric mass ratio (default: 0.25)

    Returns:
        float or np.ndarray: Chirp mass in solar masses producing the given
        number of precession cycles for the provided omega_tilde.
    """
    n_cycles = np.asarray(n_cycles, dtype=float)
    omega_tilde = np.asarray(omega_tilde, dtype=float)

    # Constants derived from number_of_prec_cycles
    K = (6 ** (3.0 / 2.0)) * np.pi * SOLMASS2SEC / (eta ** (3.0 / 5.0))
    C = ((5000.0 / 96.0) * (6 ** (5.0 / 2.0))) / (
        2.0 * (np.pi**2) * (eta ** (2.0 / 5.0))
    )

    # Avoid division by zero for omega_tilde = 0 (no precession => n_cycles should be 0)
    if np.any(omega_tilde == 0):
        raise ValueError("omega_tilde must be non-zero to invert n_prec → mcz")

    denom = f_min * (n_cycles / (C * omega_tilde) + K)
    if np.any(denom <= 0):
        raise ValueError(
            "Requested n_cycles and omega_tilde lead to non-physical mcz (denominator ≤ 0)"
        )

    mcz_msun = 1.0 / denom
    return mcz_msun


def get_lens_limits_for_RP_L(
    mcz_msun: float,
    omega_tilde: float,
    lower: Union[str, float] = "min",
    upper: Union[str, float] = "max",
    y: float = 0.25,
    f_min: float = 20,
    eta: float = 0.25,
) -> dict:
    """
    Calculates the lower and upper limits of the lens mass [solar mass] and time delay [second] such that the number of modulation cycles in the lensed waveform is comparable to the number of precession cycles.

    Args:
        mcz_msun (float): Chirp mass [solar mass].
        omega_tilde (float): Precession amplitude parameter [dimensionless].
        lower (str or float, optional): The lower limit of the number of modulation cycles in the lensed waveform. Default is "min" for boundary between wave optics and geometric optics.
        upper (str or float, optional): The upper limit of the number of modulation cycles in the lensed waveform. Default is "max" for matching the number of precession cycles.
        y (float, optional): The source position of the lens [dimensionless]. Default is 0.25.
        f_min (float, optional): The minimum frequency [Hz]. Default is 20 Hz.
        eta (float, optional): Symmetric mass ratio [dimensionless]. Default is 0.25.

    Returns:
        dict: A dictionary containing the following keys:
        - "MLz_min" (float): The minimum lens mass [solar mass].
        - "MLz_max" (float): The maximum lens mass [solar mass].
        - "td_min" (float): The minimum time delay [second].
        - "td_max" (float): The maximum time delay [second].
    """

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

    results = {
        "MLz_min": MLz_min,
        "MLz_max": MLz_max,
        "td_min": td_min,
        "td_max": td_max,
    }

    return results


def pickle_data(data, dir: str, filename: str) -> str:
    """
    Pickles the given data and saves it as a file with the specified filename.

    Args:
        data: The data to be pickled.
        dir (str): The directory to save the pickled data.
        filename (str): The name of the file to save the pickled data.

    Returns:
        str: The filepath of the saved file.
    """
    now = datetime.now()
    filename = filename + "_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".pkl"
    filepath = os.path.join(dir, filename)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
        print("Pickle saved as", filepath)
    return filepath


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            end_time = time.time()
            total_time = end_time - start_time
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(
                f"Total time to run the script: {int(hours)}:{int(minutes)}:{round(seconds, 2)} (h:m:s)"
            )

    return wrapper


############################
# Section 3: Data Handling #
############################


def omit_numerical_errors(
    arr: Union[np.ndarray, List, Tuple], n: int = 16, order: float = 1.5
) -> np.ndarray:
    """
    Omits numerical errors in an array by replacing values that are greater than a certain order of the median with NaN.

    Uses a rolling median filter for efficiency. The function handles edge cases by extending the array
    with reflected values or using available neighbors.

    Args:
        arr (array-like): The input array.
        n (int): The number of neighbors to consider when calculating the median. Must be odd and >= 3.
        order (float): The order of the median used to determine if a value is an error. Default is 1.5.

    Returns:
        np.ndarray: The modified array with numerical errors omitted.

    Raises:
        ValueError: If n is even or less than 3.
        ValueError: If array length is less than n.
    """
    # Input validation
    if n % 2 == 0:
        raise ValueError("n must be odd for symmetric neighbor selection")
    if n < 3:
        raise ValueError("n must be at least 3")

    arr_copy = np.asarray(arr, dtype=float)
    if len(arr_copy) < n:
        raise ValueError(f"Array length {len(arr_copy)} must be >= n={n}")

    # Use scipy's median filter for efficiency (much faster than Python loops)
    try:
        from scipy.signal import medfilt

        # medfilt requires odd kernel size
        median_values = medfilt(arr_copy, kernel_size=n)
    except ImportError:
        # Fallback to manual rolling median if scipy not available
        median_values = _rolling_median_fallback(arr_copy, n)

    # Replace outliers with NaN
    threshold = order * median_values
    arr_copy[arr_copy > threshold] = np.nan

    return arr_copy


def _rolling_median_fallback(arr: np.ndarray, n: int) -> np.ndarray:
    """
    Fallback implementation of rolling median when scipy is not available.
    This is slower than scipy.medfilt but provides the same functionality.

    Args:
        arr: Input array
        n: Window size (must be odd)

    Returns:
        Array of rolling medians
    """
    half_window = n // 2
    result = np.empty_like(arr)

    for i in range(len(arr)):
        start = max(0, i - half_window)
        end = min(len(arr), i + half_window + 1)
        window = arr[start:end]
        result[i] = np.nanmedian(window)

    return result


###########################################
# Section 4: Inclination & Special Coords #
###########################################


def calculate_cosJN_params(params: dict) -> float:
    """
    Calculates the cosine of the angle between the total angular momentum (J) of the BBH system and the line of sight (N).

    Args:
        params (dict): A dictionary containing the parameters the following keys:
        - "phi_S" (float): The azimuthal angle of N with respect to the detector frame.
        - "theta_S" (float): The polar angle of N with respect to the detector frame.
        - "phi_J" (float): The azimuthal angle of J with respect to the detector frame.
        - "theta_J" (float): The polar angle of J with respect to the detector frame.

    Returns:
        float: The cosine of the angle between J and N.
    """
    return np.sin(params["theta_J"]) * np.sin(params["theta_S"]) * np.cos(
        params["phi_J"] - params["phi_S"]
    ) + np.cos(params["theta_J"]) * np.cos(params["theta_S"])


def calculate_cosJN(
    phi_S: Union[float, np.ndarray],
    theta_S: Union[float, np.ndarray],
    phi_J: Union[float, np.ndarray],
    theta_J: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculates the cosine of the angle between the total angular momentum (J) of the BBH system and the line of sight (N).

    Args:
        phi_S (float): The azimuthal angle of N with respect to the detector frame.
        theta_S (float): The polar angle of N with respect to the detector frame.
        phi_J (float): The azimuthal angle of J with respect to the detector frame.
        theta_J (float): The polar angle of J with respect to the detector frame.

    Returns:
        float: The cosine of the angle between J and N.
    """
    print("order of arguments: phi_S, theta_S, phi_J, theta_J")
    return np.sin(theta_J) * np.sin(theta_S) * np.cos(phi_J - phi_S) + np.cos(
        theta_J
    ) * np.cos(theta_S)


def find_FaceOn_coords(
    fix: str, fixed_phi: float, fixed_theta: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the coordinate values where the BBH source is face-on (|cos(JN)| = 1 within error).

    Args:
        fix (str): The parameter to fix. Either "S" for fixing the source's sky location or "J" for fixing the binary orientation.
        fixed_phi (float): The fixed azimuthal angle.
        fixed_theta (float): The fixed polar angle.

    Returns:
        tuple: A tuple containing the azimuthal and polar angles where the source is face-on.
    """
    n_pts = 151
    phi_arr = np.linspace(0, 2 * np.pi, n_pts)
    theta_arr = np.linspace(0, np.pi, n_pts)
    X, Y = np.meshgrid(phi_arr, theta_arr)

    if fix == "S":
        Z = calculate_cosJN(fixed_phi, fixed_theta, X, Y)
    else:  # fix == 'J'
        Z = calculate_cosJN(X, Y, fixed_phi, fixed_theta)

    # condition where |Z| = 1 within error
    cond = np.isclose(np.abs(Z), 1, rtol=0, atol=1e-3)
    return X[cond], Y[cond]


def find_EdgeOn_coords(
    fix: str, fixed_phi: float, fixed_theta: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the coordinate values where the BBH source is edge-on (|cos(JN)| = 0 within error).

    Args:
        fix (str): The parameter to fix. Either "S" for fixing the source's sky location or "J" for fixing the binary orientation.
        fixed_phi (float): The fixed azimuthal angle.
        fixed_theta (float): The fixed polar angle.

    Returns:
        tuple: A tuple containing the azimuthal and polar angles where the source is edge-on.
    """
    n_pts = 151
    phi_arr = np.linspace(0, 2 * np.pi, n_pts)
    theta_arr = np.linspace(0, np.pi, n_pts)
    X, Y = np.meshgrid(phi_arr, theta_arr)

    if fix == "S":
        Z = calculate_cosJN(fixed_phi, fixed_theta, X, Y)
    else:  # fix == 'J'
        Z = calculate_cosJN(X, Y, fixed_phi, fixed_theta)

    # condition where |Z| = 0 within error
    cond = np.isclose(np.abs(Z), 0, rtol=0, atol=1e-2)
    return X[cond], Y[cond]


##################
# Section 5: SNR #
##################


def Sn(
    f_arr: np.ndarray,
    f_min: float = 20,
    delta_f: float = 0.25,
    frequencySeries=True,
) -> Union[np.ndarray, FrequencySeries]:
    """
    Calculates the power spectral density of the aLIGO noise curve based on arXiv:0903.0338.

    Parameters
    ----------
    f_arr : np.ndarray
        The frequency array.
    f_min : float, optional
        The minimum frequency. Defaults to 20 Hz.
    delta_f : float, optional
        The frequency step size. Defaults to 0.25 Hz.
    frequencySeries : bool, optional
        If True, returns a FrequencySeries object. Defaults to True.

    Returns
    -------
    np.ndarray or FrequencySeries
        The power spectral density of the aLIGO noise curve.
    """

    # Vectorized implementation of the aLIGO PSD (0903.0338)
    f_arr = np.asarray(f_arr)
    S0 = 1e-49
    f0 = 215.0
    x = f_arr / f0

    # Base PSD expression evaluated elementwise
    Sn_temp = (
        np.power(x, -4.14)
        - 5.0 * np.power(x, -2.0)
        + 111.0 * ((1.0 - x**2 + 0.5 * x**4) / (1.0 + 0.5 * x**2))
    )
    Sn_val = Sn_temp * S0

    # Enforce infinite PSD below f_min
    Sn_val = np.where(f_arr < f_min, np.inf, Sn_val)

    if frequencySeries:
        return FrequencySeries(Sn_val, delta_f=delta_f)
    return Sn_val


def SNR(
    params: dict,
    f_min: float = 20,
    delta_f: float = 0.25,
    psd: FrequencySeries = None,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
) -> float:
    """
    Calculates the Signal-to-Noise Ratio (SNR) for a given dictionary of parameters.

    Parameters
    ----------
    params : dict
        A dictionary containing the parameters.
    f_min : float, optional
        The lower frequency cutoff. Defaults to 20 Hz.
    delta_f : float, optional
        The frequency step size. Defaults to 0.25 Hz.
    psd : FrequencySeries, optional
        The power spectral density of the detector noise. If not provided, it will be calculated based on the aLIGO noise curve from arXiv:0903.0338, as a function of the source waveform's frequency range. Defaults to None.
    lens_Class : class, optional
        The class to use for lensing parameters. Defaults to LensingGeo.
    prec_Class : class, optional
        The class to use for precessing parameters. Defaults to Precessing.

    Returns
    -------
    float
        The calculated SNR value.
    """
    if "MLz" in params and "y" in params:  # lensing parameters, use lens_Class
        gw_inst = lens_Class(params)
    else:  # precessing parameters, use prec_Class
        gw_inst = prec_Class(params)

    f_cut = gw_inst.f_cut()
    f_arr = np.arange(f_min, f_cut, delta_f)
    if psd is None:
        psd = Sn(f_arr, f_min=f_min, delta_f=delta_f)
    h = gw_inst.strain(f_arr, delta_f=delta_f)

    # calculate SNR
    integrand = np.abs(h) ** 2 / psd
    integrated_inner_product = simpson(integrand, x=f_arr)
    snr = np.sqrt(4 * np.real(integrated_inner_product))

    return snr


#######################
# Section 6: Mismatch #
#######################


def optimized_match_bounded(
    vec1,
    vec2,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    v1_norm=None,
    v2_norm=None,
    return_phase=False,
):
    """Compatibility wrapper; canonical implementation lives in modules.match_utils."""
    from modules.match_utils import optimized_match_bounded as _optimized_match_bounded

    return _optimized_match_bounded(
        vec1,
        vec2,
        psd=psd,
        low_frequency_cutoff=low_frequency_cutoff,
        high_frequency_cutoff=high_frequency_cutoff,
        v1_norm=v1_norm,
        v2_norm=v2_norm,
        return_phase=return_phase,
    )


def mismatch_from_params(
    t_params: dict,  # template parameters
    s_params: dict,  # source parameters
    f_min: float = 20,
    delta_f: float = 0.25,
    psd: FrequencySeries = None,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_opt_match=True,
    compare_both=False,
) -> dict:
    """Compatibility wrapper; canonical implementation lives in modules.match_utils."""
    from modules.match_utils import mismatch_from_params as _mismatch_from_params

    return _mismatch_from_params(
        t_params,
        s_params,
        f_min=f_min,
        delta_f=delta_f,
        psd=psd,
        lens_Class=lens_Class,
        prec_Class=prec_Class,
        use_opt_match=use_opt_match,
        compare_both=compare_both,
        get_gw_func=get_gw,
        sn_func=Sn,
    )


def mismatch_from_strains(
    t_strain: Union[np.ndarray, FrequencySeries],
    s_strain: Union[np.ndarray, FrequencySeries],
    f_min: float = 20,
    delta_f: float = 0.25,
    psd: FrequencySeries = None,
    use_opt_match=True,
    compare_both=False,
) -> dict:
    """Compatibility wrapper; canonical implementation lives in modules.match_utils."""
    from modules.match_utils import mismatch_from_strains as _mismatch_from_strains

    return _mismatch_from_strains(
        t_strain,
        s_strain,
        f_min=f_min,
        delta_f=delta_f,
        psd=psd,
        use_opt_match=use_opt_match,
        compare_both=compare_both,
        sn_func=Sn,
    )


################################################
# Section 7: Optimize Mismatch Over Parameters #
################################################


def optimize_mismatch_mcz(
    t_params: dict,  # template parameters
    s_params: dict,  # source parameters
    f_min: float = 20,
    delta_f: float = 0.25,
    psd: FrequencySeries = None,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_opt_match=True,
    compare_both=False,
) -> dict:
    """Compatibility wrapper; canonical implementation lives in modules.match_utils."""
    from modules.match_utils import optimize_mismatch_mcz as _optimize_mismatch_mcz

    return _optimize_mismatch_mcz(
        t_params,
        s_params,
        f_min=f_min,
        delta_f=delta_f,
        psd=psd,
        lens_Class=lens_Class,
        prec_Class=prec_Class,
        use_opt_match=use_opt_match,
        compare_both=compare_both,
        set_to_params_func=set_to_params,
        get_gw_func=get_gw,
        sn_func=Sn,
        mismatch_from_strains_func=mismatch_from_strains,
        solmass2sec=SOLMASS2SEC,
    )


def optimize_mismatch_gammaP(
    t_params: dict,  # template parameters
    s_params: dict = None,  # source parameters (ignored if s_strain provided)
    s_strain: FrequencySeries = None,  # precomputed source strain
    f_min: float = 20,
    delta_f: float = 0.25,
    psd: FrequencySeries = None,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_opt_match=True,
    compare_both=False,
    grid_points: int = 101,
    gamma_grid: np.ndarray = None,
    two_stage=False,
    coarse_points: int = 17,
    xatol: float = 1e-3,
    maxiter: int = 50,
) -> dict:
    """Compatibility wrapper; canonical implementation lives in modules.match_utils."""
    from modules.match_utils import (
        optimize_mismatch_gammaP as _optimize_mismatch_gammaP,
    )

    return _optimize_mismatch_gammaP(
        t_params,
        s_params=s_params,
        s_strain=s_strain,
        f_min=f_min,
        delta_f=delta_f,
        psd=psd,
        lens_Class=lens_Class,
        prec_Class=prec_Class,
        use_opt_match=use_opt_match,
        compare_both=compare_both,
        grid_points=grid_points,
        gamma_grid=gamma_grid,
        two_stage=two_stage,
        coarse_points=coarse_points,
        xatol=xatol,
        maxiter=maxiter,
        set_to_params_func=set_to_params,
        get_gw_func=get_gw,
        sn_func=Sn,
        mismatch_from_strains_func=mismatch_from_strains,
    )


def find_optimized_coalescence_params(
    t_params: dict,  # template parameters
    s_params: dict,  # source parameters
    f_min: float = 20,
    delta_f: float = 0.25,
    psd: FrequencySeries = None,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_opt_match=True,
    compare_both=False,
    optimize_gammaP=True,
    verify_optimization=False,
    **kwargs,
) -> dict:
    """Compatibility wrapper; canonical implementation lives in modules.match_utils."""
    from modules.match_utils import (
        find_optimized_coalescence_params as _find_optimized_coalescence_params,
    )

    return _find_optimized_coalescence_params(
        t_params,
        s_params,
        f_min=f_min,
        delta_f=delta_f,
        psd=psd,
        lens_Class=lens_Class,
        prec_Class=prec_Class,
        use_opt_match=use_opt_match,
        compare_both=compare_both,
        optimize_gammaP=optimize_gammaP,
        verify_optimization=verify_optimization,
        set_to_params_func=set_to_params,
        get_fcut_from_mcz_func=get_fcut_from_mcz,
        sn_func=Sn,
        optimize_mismatch_gammaP_func=optimize_mismatch_gammaP,
        mismatch_from_params_func=mismatch_from_params,
        get_gw_func=get_gw,
        solmass2sec=SOLMASS2SEC,
        **kwargs,
    )
