#############################
# Section 1: Import Modules #
#############################


# import py scripts
from modules.Classes_ver2 import *
from modules.default_params_ver1 import *
from modules.functions_ver2 import *
from modules.contours_ver1 import *


#######################
# Section: Error Bars #
#######################

from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def polynomial(x, a, b, c):
    return a * x**2 + b * x + c


def gaussian(x, a, mu, sigma):
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def get_error_bars_ver1(X, Y, Z, min_idx):
    """
    Calculate the FWHM of the dip in X and Y.

    Parameters
    ----------
    X : 2D array
        X values.
    Y : 2D array
        Y values.
    Z : 2D array
        Z values.
    min_idx : tuple
        Index of the minimum value in Z.
    thres_factor : float
        Factor to multiply with Z[min_idx] to get the threshold.
    thres_diff : float
        Value to add to the threshold.

    Returns
    -------
    X_err : 2D array
        FWHM of the dip in X for each row and column.
    Y_err : 2D array
        FWHM of the dip in Y for each row and column.
    """

    omega_arr = X[min_idx[0], :]
    ep_omega_arr = Z[min_idx[0], :]
    popt_omega, pcov_omega = curve_fit(polynomial, omega_arr, ep_omega_arr)

    theta_arr = Y[:, min_idx[1]]
    ep_theta_arr = Z[:, min_idx[1]]
    popt_theta, pcov_theta = curve_fit(polynomial, theta_arr, ep_theta_arr)

    return omega_err, theta_err


def contour_stats_ver1(X, Y, Z, g_min_mtx) -> dict:
    min_idx = np.unravel_index(np.argmin(Z, axis=None), Z.shape)
    max_idx = np.unravel_index(np.argmax(Z, axis=None), Z.shape)

    X_err, Y_err = get_error_bars_ver1(X, Y, Z, min_idx)

    results = {
        "ep_0_0": Z[0, 0],
        "ep_min": Z[min_idx],
        "ep_min_omega_tilde": X[min_idx],
        "ep_min_theta_tilde": Y[min_idx],
        "ep_min_gammaP": g_min_mtx[min_idx],
        "ep_max": Z[max_idx],
        "ep_max_omega_tilde": X[max_idx],
        "ep_max_theta_tilde": Y[max_idx],
        "ep_max_gammaP": g_min_mtx[max_idx],
        "ep_max_min_ratio": Z[max_idx] / Z[min_idx],
        "omega_tilde_err": X_err,
        "theta_tilde_err": Y_err,
    }

    return results


def get_contours_stats_ver1(d: dict) -> dict:
    d_copy = copy.deepcopy(d)
    for k in d_copy.keys():
        if isinstance(k, str):
            continue
        contour_data = d_copy[k]["contour"]
        omega_mtx = contour_data["omega_matrix"]
        theta_mtx = contour_data["theta_matrix"]
        ep_mtx = contour_data["epsilon_matrix"]
        g_mtx = contour_data["gammaP_min_matrix"]
        d_copy[k]["stats"] = contour_stats_ver1(omega_mtx, theta_mtx, ep_mtx, g_mtx)

    return d_copy


############################################
# Section 2: NP vs Lensed Mismatch Contour #
############################################


def mismatch_contour_NP_L(t_params: dict, s_params: dict):
    t_params_copy, s_params_copy = set_to_params(t_params, s_params)
    results = optimize_mismatch_gammaP(t_params_copy, s_params_copy)
    return results["ep_min"]


##################################################
# Section 4: Dictionary of NP vs Lensed Contours #
##################################################


def create_mismatch_contours_td_NP_L(
    t_params: dict, s_params: dict, MLz_arr: np.ndarray
) -> dict:
    I = LensingGeo(s_params).I()
    td_arr = np.zeros_like(MLz_arr)
    results = {}

    for i, MLz in enumerate(MLz_arr):
        s_params["MLz"] = MLz * solar_mass
        td = LensingGeo(s_params).td()
        td = round(td, 6)  # Round to 6 decimal places
        td_arr[i] = td
        results[td] = {}
        results[td]["epsilon"] = mismatch_contour_NP_L(t_params, s_params)

    results["source_params"] = s_params
    results["I"] = I
    results["td_arr"] = td_arr
    results["MLz_arr"] = MLz_arr

    return results


def create_mismatch_contours_I_NP_L(
    t_params: dict, s_params: dict, td: float, y_arr: np.ndarray
) -> dict:
    # create MLz_arr from y_arr based on the same time delay
    MLz_arr = get_MLz_from_td(td, y_arr)
    I_arr = np.zeros_like(MLz_arr)
    results = {}

    for i in range(len(y_arr)):
        s_params["y"] = y_arr[i]
        s_params["MLz"] = MLz_arr[i] * solar_mass
        I = LensingGeo(s_params).I()
        I = round(I, 6)  # Round to 6 decimal places
        I_arr[i] = I
        results[I] = {}
        results[I]["epsilon"] = mismatch_contour_NP_L(t_params, s_params)

    results["source_params"] = s_params
    results["td"] = td
    results["I_arr"] = I_arr
    results["y_arr"] = y_arr
    results["MLz_arr"] = MLz_arr

    return results


#############################
# Section 7: Super Contours #
#############################


def get_super_contour(t_params, s_params, td_arr, y_arr):
    results = {}
    for td in td_arr:
        td = round(td, 6)
        results[td] = create_mismatch_contours_I(t_params, s_params, td, y_arr)

    return results


def get_super_contour_stats(d: dict, thres_factor=1.01, thres_diff=0.0) -> dict:
    d_copy = copy.deepcopy(d)
    for k in d_copy.keys():
        if isinstance(k, str):
            continue
        d_copy[k] = get_contours_stats(d_copy[k], thres_factor, thres_diff)

    return d_copy


def get_super_contour_stats_ver1(d: dict) -> dict:
    d_copy = copy.deepcopy(d)
    for k in d_copy.keys():
        if isinstance(k, str):
            continue
        d_copy[k] = get_contours_stats_ver1(d_copy[k])

    return d_copy


def get_super_contour_NP_L(t_params, s_params, td_arr, y_arr):
    results = {}
    for td in td_arr:
        td = round(td, 6)
        results[td] = create_mismatch_contours_I_NP_L(t_params, s_params, td, y_arr)

    return results
