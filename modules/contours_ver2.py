#############################
# Section 1: Import Modules #
#############################


# import py scripts
from modules.functions_ver2 import *

# import libraries
from multiprocessing import Pool, cpu_count


#################################
# Section 2: A Mismatch Contour #
#################################


def compute_mismatch(
    t_params: dict, s_params: dict, X: np.ndarray, Y: np.ndarray, r, c
) -> tuple:
    t_params_copy, s_params_copy = set_to_params(t_params, s_params)

    t_params_copy["omega_tilde"] = X[r, c]
    t_params_copy["theta_tilde"] = Y[r, c]

    results = optimize_mismatch_gammaP(t_params_copy, s_params_copy)

    return (results["ep_min"], results["ep_min_gammaP"])


def create_mismatch_contour_parallel(t_params: dict, s_params: dict) -> dict:
    nx_pts = 41
    ny_pts = 151
    omega_arr = np.linspace(0, 4, nx_pts)
    theta_arr = np.linspace(0, 15, ny_pts)
    X, Y = np.meshgrid(omega_arr, theta_arr)
    Z = np.zeros_like(X)
    g_min_mtx = np.zeros_like(X)

    # Create a pool of worker processes
    with Pool(cpu_count()) as pool:  # Use maximum number of cores
        results = []
        for r in range(ny_pts):
            for c in range(nx_pts):
                results.append(
                    pool.apply_async(
                        compute_mismatch, args=(t_params, s_params, X, Y, r, c)
                    )
                )

        for r in range(ny_pts):
            for c in range(nx_pts):
                idx = r * nx_pts + c
                Z[r, c], g_min_mtx[r, c] = results[idx].get()

    results = {
        "omega_matrix": X,
        "theta_matrix": Y,
        "epsilon_matrix": Z,
        "gammaP_min_matrix": g_min_mtx,
        "source_params": s_params,
        "template_params": t_params,
    }

    return results


def compute_mismatch_L_NP(t_params: dict, s_params: dict) -> dict:
    t_params_copy, s_params_copy = set_to_params(t_params, s_params)
    results = mismatch(t_params_copy, s_params_copy)
    return {"epsilon": results["mismatch"], "source_params": s_params}


#####################################
# Section 3: Dictionary of Contours #
#####################################


def create_contours_td(
    t_params: dict, s_params: dict, I: float, td_arr: np.ndarray, what_template="RP"
) -> dict:

    I = np.round(I, 6)
    td_arr = np.round(td_arr, 6)
    y = get_y_from_I(I)
    MLz_arr = get_MLz_from_td(td_arr, y)
    results = {}

    for i in range(len(td_arr)):
        s_params["y"] = y
        s_params["MLz"] = MLz_arr[i] * solar_mass
        td = td_arr[i]
        results[td] = {}
        if what_template == "RP":
            results[td]["contour"] = create_mismatch_contour_parallel(
                t_params, s_params
            )
        elif what_template == "NP":
            results[td]["contour"] = compute_mismatch_L_NP(t_params, s_params)

    results["I"] = I
    results["td_arr"] = td_arr
    results["y"] = y
    results["MLz_arr"] = MLz_arr

    return results


def create_contours_I(
    t_params: dict, s_params: dict, td: float, I_arr: np.ndarray, what_template="RP"
) -> dict:

    td = np.round(td, 6)
    I_arr = np.round(I_arr, 6)
    y_arr = get_y_from_I(I_arr)
    MLz_arr = get_MLz_from_td(td, y_arr)
    results = {}

    for i in range(len(I_arr)):
        s_params["y"] = y_arr[i]
        s_params["MLz"] = MLz_arr[i] * solar_mass
        I = I_arr[i]
        results[I] = {}
        if what_template == "RP":
            results[I]["contour"] = create_mismatch_contour_parallel(t_params, s_params)
        elif what_template == "NP":
            results[I]["contour"] = compute_mismatch_L_NP(t_params, s_params)

    results["td"] = td
    results["I_arr"] = I_arr
    results["y_arr"] = y_arr
    results["MLz_arr"] = MLz_arr

    return results


############################
# Section 4: Super Contour #
############################


def create_super_contour(
    t_params: dict,
    s_params: dict,
    td_arr: np.ndarray,
    I_arr: np.ndarray,
    what_template="RP",
) -> dict:

    td_arr = np.round(td_arr, 6)
    I_arr = np.round(I_arr, 6)
    results = {}

    for td in td_arr:
        results[td] = create_contours_I(t_params, s_params, td, I_arr, what_template)

    results["td_arr"] = td_arr
    results["I_arr"] = I_arr
    results["source_params"] = s_params  # for convenience
    results["template_params"] = t_params  # for convenience

    return results


##############################
# Section 5: Post-Processing #
##############################


def get_error_bars(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    min_idx: tuple,
    thres_factor: float,
    thres_diff: float,
) -> tuple:
    """
    Get the X and Y values where Z is less (or greater) than a threshold.

    Parameters
    ----------
    X : numpy.ndarray
        Array of X values.
    Y : numpy.ndarray
        Array of Y values.
    Z : numpy.ndarray
        Array of Z values.
    min_idx : tuple
        Index of the minimum value in Z.
    thres_factor : float
        Factor to multiply with Z[min_idx] to get the threshold.
    thres_diff : float
        Value to add to the threshold.

    Returns
    -------
    X_vals : numpy.ndarray
        Array of X values where Z is less (or greater) than the threshold.
    Y_vals : numpy.ndarray
        Array of Y values where Z is less (or greater) than the threshold.
    """
    Z_thres = Z[min_idx] * thres_factor + thres_diff
    indices = np.where(Z < Z_thres)
    X_vals = X[indices]
    Y_vals = Y[indices]
    return X_vals, Y_vals


def get_indiv_contour_stats(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    g_min_mtx: np.ndarray,
    thres_factor: float,
    thres_diff: float,
) -> dict:
    min_idx = np.unravel_index(np.argmin(Z, axis=None), Z.shape)
    max_idx = np.unravel_index(np.argmax(Z, axis=None), Z.shape)

    X_err, Y_err = get_error_bars(X, Y, Z, min_idx, thres_factor, thres_diff)

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


def get_contours_stats(d: dict, thres_factor=1.01, thres_diff=0.0) -> dict:
    d_copy = copy.deepcopy(d)
    for k in d_copy.keys():
        if isinstance(k, str):
            continue
        contour_data = d_copy[k]["contour"]
        omega_mtx = contour_data["omega_matrix"]
        theta_mtx = contour_data["theta_matrix"]
        ep_mtx = contour_data["epsilon_matrix"]
        g_mtx = contour_data["gammaP_min_matrix"]
        d_copy[k]["stats"] = get_indiv_contour_stats(
            omega_mtx, theta_mtx, ep_mtx, g_mtx, thres_factor, thres_diff
        )

    return d_copy


def get_asym_err(d: dict, k_arr: np.ndarray, param_name: str) -> list:
    if param_name == "omega_tilde":
        param_arr = np.array([d[k]["stats"]["ep_min_omega_tilde"] for k in k_arr])
        lower_err = param_arr - np.array(
            [np.min(d[k]["stats"]["omega_tilde_err"]) for k in k_arr]
        )
        upper_err = (
            np.array([np.max(d[k]["stats"]["omega_tilde_err"]) for k in k_arr])
            - param_arr
        )

    elif param_name == "theta_tilde":
        param_arr = np.array([d[k]["stats"]["ep_min_theta_tilde"] for k in k_arr])
        lower_err = param_arr - np.array(
            [np.min(d[k]["stats"]["theta_tilde_err"]) for k in k_arr]
        )
        upper_err = (
            np.array([np.max(d[k]["stats"]["theta_tilde_err"]) for k in k_arr])
            - param_arr
        )

    asym_err = [lower_err, upper_err]

    return asym_err


def get_super_contour_stats(d: dict, thres_factor=1.01, thres_diff=0.0) -> dict:
    d_copy = copy.deepcopy(d)
    for k in d_copy.keys():
        if isinstance(k, str):
            continue
        d_copy[k] = get_contours_stats(d_copy[k], thres_factor, thres_diff)

    return d_copy
