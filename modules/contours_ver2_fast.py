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
    nx_pts = 21
    ny_pts = 76
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
    results = optimize_mismatch_gammaP(t_params_copy, s_params_copy)
    return {"epsilon": results["ep_min"], "source_params": s_params}


#####################################
# Section 3: Dictionary of Contours #
#####################################


def create_contours_td(
    t_params: dict, s_params: dict, I: float, td_arr: np.ndarray, what_template="RP"
) -> dict:
    y = get_y_from_I(I)
    MLz_arr = get_MLz_from_td(td_arr, y)
    results = {}

    for i in range(len(td_arr)):
        s_params["y"] = y
        s_params["MLz"] = MLz_arr[i] * solar_mass
        td = td_arr[i]
        td = round(td, 6)  # Round to 6 decimal places
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
    y_arr = get_y_from_I(I_arr)
    MLz_arr = get_MLz_from_td(td, y_arr)
    results = {}

    for i in range(len(I_arr)):
        s_params["y"] = y_arr[i]
        s_params["MLz"] = MLz_arr[i] * solar_mass
        I = I_arr[i]
        I = round(I, 6)  # Round to 6 decimal places
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
    results = {}
    for td in td_arr:
        td = round(td, 6)
        results[td] = create_contours_I(t_params, s_params, td, I_arr, what_template)

    results["td_arr"] = td_arr
    results["I_arr"] = I_arr
    results["source_params"] = s_params  # for convenience
    results["template_params"] = t_params  # for convenience
    return results
