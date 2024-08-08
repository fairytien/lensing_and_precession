#############################
# Section 1: Import Modules #
#############################


# import py scripts
from modules.functions_ver2 import *

# import libraries
from multiprocessing import Pool, cpu_count


###############################
# Section 2: RP Template Bank #
###############################


def compute_RP_template(
    t_params: dict,
    omega_grid: np.ndarray,
    theta_grid: np.ndarray,
    gamma_grid: np.ndarray,
    idx: tuple,
    **kwargs,
) -> tuple[tuple, np.ndarray]:

    # Compute the template based on t_params
    t_params_copy = copy.deepcopy(t_params)
    t_params_copy["omega_tilde"] = omega_grid[idx]
    t_params_copy["theta_tilde"] = theta_grid[idx]
    t_params_copy["gamma_P"] = gamma_grid[idx]

    template = get_gw(t_params_copy, frequencySeries=False, **kwargs)["strain"]

    return idx, template


def create_RP_templates(t_params: dict, filename: str, npz=True) -> np.ndarray:
    nx_pts = 41
    ny_pts = 151
    nz_pts = 51
    omega_arr = np.linspace(0, 4, nx_pts)
    theta_arr = np.linspace(0, 15, ny_pts)
    gamma_arr = np.linspace(0, 2 * np.pi, nz_pts)

    # Create a 3D meshgrid
    omega_grid, theta_grid, gamma_grid = np.meshgrid(
        omega_arr, theta_arr, gamma_arr, indexing="ij"
    )

    # Initialize an empty grid to store the templates
    template_grid = np.empty((nx_pts, ny_pts, nz_pts), dtype=object)

    # Create a list of indices to parallelize the computation
    idx_list = list(np.ndindex(nx_pts, ny_pts, nz_pts))

    # Use Pool to parallelize the computation
    with Pool(cpu_count()) as pool:  # Use maximum number of cores
        results = pool.starmap(
            compute_RP_template,
            [(t_params, omega_grid, theta_grid, gamma_grid, idx) for idx in idx_list],
        )
    # Store the results in the template grid
    for idx, template in results:
        template_grid[idx] = template

    if npz:
        np.savez_compressed(
            filename, template_grid=template_grid, template_params=t_params
        )

    return template_grid


#################################
# Section 3: A Mismatch Contour #
#################################


def compute_mismatch(
    t_strain: Union[np.ndarray, FrequencySeries],
    s_strain: Union[np.ndarray, FrequencySeries],
    f_min=20,
    delta_f=0.25,
    psd=None,
    use_opt_match=True,
) -> dict:
    if not isinstance(t_strain, FrequencySeries):
        t_strain = FrequencySeries(t_strain, delta_f)
    if not isinstance(s_strain, FrequencySeries):
        s_strain = FrequencySeries(s_strain, delta_f)

    # Get the psd from s_strain; Should provide psd to avoid recomputing it and save time
    if psd is None:
        f_arr = s_strain.sample_frequencies + f_min
        psd = Sn(f_arr)

    match_func = optimized_match if use_opt_match else match
    match_val, index, phi = match_func(t_strain, s_strain, psd, return_phase=True)  # type: ignore
    mismatch = 1 - match_val

    return {"mismatch": mismatch, "index": index, "phi": phi}


def create_mismatch_contour(
    template_grid: np.ndarray,
    s_params: dict,
    f_min=20,
    delta_f=0.25,
    psd=None,
    use_opt_match=True,
) -> dict:
    # Get the strain from the source parameters
    s_strain = get_gw(s_params, f_min, delta_f)["strain"]

    # Get the psd from s_strain; Should provide psd to avoid recomputing it and save time
    if psd is None:
        f_arr = s_strain.sample_frequencies + f_min
        psd = Sn(f_arr)

    # Compute the mismatches in parallel
    with Pool(cpu_count()) as pool:
        results = pool.starmap(
            compute_mismatch,
            [
                (t_strain, s_strain, f_min, delta_f, psd, use_opt_match)
                for t_strain in template_grid.flatten()
            ],
        )

    # Initialize arrays and grids
    nx_pts = 41
    ny_pts = 151
    nz_pts = 51
    omega_arr = np.linspace(0, 4, nx_pts)
    theta_arr = np.linspace(0, 15, ny_pts)
    gamma_arr = np.linspace(0, 2 * np.pi, nz_pts)

    ep_grid_3D = np.zeros((nx_pts, ny_pts, nz_pts))

    omega_grid_2D, theta_grid_2D = np.meshgrid(omega_arr, theta_arr, indexing="ij")
    g_min_grid_2D, ep_grid_2D = np.zeros((nx_pts, ny_pts)), np.zeros((nx_pts, ny_pts))

    # Populate 3D grids with mismatch results
    for i, result in enumerate(results):
        i_3D = np.unravel_index(i, template_grid.shape)
        ep_grid_3D[i_3D] = result["mismatch"]

    # Find the gamma_P that minimizes the mismatch for each pair of omega_tilde and theta_tilde
    for i, omega in enumerate(omega_arr):
        for j, theta in enumerate(theta_arr):
            gamma_min_idx = np.argmin(ep_grid_3D[i, j, :])
            g_min_grid_2D[i, j] = gamma_arr[gamma_min_idx]
            ep_grid_2D[i, j] = ep_grid_3D[i, j, gamma_min_idx]

    return {
        "omega_grid": omega_grid_2D,
        "theta_grid": theta_grid_2D,
        "epsilon_grid": ep_grid_2D,
        "gamma_min_grid": g_min_grid_2D,
        "source_params": s_params,
    }


#####################################
# Section 4: Dictionary of Contours #
#####################################


def create_contours_td(
    template_grid: np.ndarray,
    s_params: dict,
    I: float,
    td_arr: np.ndarray,
    f_min=20,
    delta_f=0.25,
    psd=None,
    what_template="RP",
) -> dict:

    if psd is None:
        f_cut = get_fcut_from_mcz(s_params["mcz"])
        f_arr = np.arange(f_min, f_cut, delta_f)
        psd = Sn(f_arr)

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
            results[td]["contour"] = create_mismatch_contour(
                template_grid, s_params, f_min, delta_f, psd
            )
        elif what_template == "NP":
            # TODO
            pass

    results["I"] = I
    results["td_arr"] = td_arr
    results["y"] = y
    results["MLz_arr"] = MLz_arr

    return results


def create_contours_I(
    template_grid: np.ndarray,
    s_params: dict,
    td: float,
    I_arr: np.ndarray,
    f_min=20,
    delta_f=0.25,
    psd=None,
    what_template="RP",
) -> dict:

    if psd is None:
        f_cut = get_fcut_from_mcz(s_params["mcz"])
        f_arr = np.arange(f_min, f_cut, delta_f)
        psd = Sn(f_arr)

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
            results[I]["contour"] = create_mismatch_contour(
                template_grid, s_params, f_min, delta_f, psd
            )
        elif what_template == "NP":
            # TODO
            pass

    results["td"] = td
    results["I_arr"] = I_arr
    results["y_arr"] = y_arr
    results["MLz_arr"] = MLz_arr

    return results


############################
# Section 5: Super Contour #
############################


def create_super_contour(
    template_grid: np.ndarray,
    s_params: dict,
    td_arr: np.ndarray,
    I_arr: np.ndarray,
    f_min=20,
    delta_f=0.25,
    psd=None,
    what_template="RP",
) -> dict:

    if psd is None:
        f_cut = get_fcut_from_mcz(s_params["mcz"])
        f_arr = np.arange(f_min, f_cut, delta_f)
        psd = Sn(f_arr)

    td_arr = np.round(td_arr, 6)
    I_arr = np.round(I_arr, 6)
    results = {}

    for td in td_arr:
        results[td] = create_contours_I(
            template_grid, s_params, td, I_arr, f_min, delta_f, psd, what_template
        )

    results["td_arr"] = td_arr
    results["I_arr"] = I_arr
    results["source_params"] = s_params  # for convenience

    return results
