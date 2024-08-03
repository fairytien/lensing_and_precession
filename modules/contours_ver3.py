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


def compute_RP_template(
    t_params: dict,
    omega_grid: np.ndarray,
    theta_grid: np.ndarray,
    gamma_grid: np.ndarray,
    idx: tuple,
) -> tuple:

    # Compute the template based on t_params
    t_params_copy = copy.deepcopy(t_params)
    t_params_copy["omega_tilde"] = omega_grid[idx]
    t_params_copy["theta_tilde"] = theta_grid[idx]
    t_params_copy["gamma_P"] = gamma_grid[idx]

    template = get_gw(t_params_copy, frequencySeries=False)["strain"]

    return idx, template


def create_RP_templates(t_params: dict) -> dict:
    nx_pts = 41
    ny_pts = 151
    nz_pts = 101
    omega_arr = np.linspace(0, 4, nx_pts)
    theta_arr = np.linspace(0, 15, ny_pts)
    gamma_arr = np.linspace(0, 2 * np.pi, nz_pts)

    # Create a 3D meshgrid
    omega_grid, theta_grid, gamma_grid = np.meshgrid(
        omega_arr, theta_arr, gamma_arr, indexing="ij"
    )

    # Initialize an empty dict to store the templates
    template_bank = {}
    template_bank["template_params"] = t_params
    template_bank["omega_array"] = omega_arr
    template_bank["theta_array"] = theta_arr
    template_bank["gamma_array"] = gamma_arr
    template_bank["omega_grid_3D"] = omega_grid
    template_bank["theta_grid_3D"] = theta_grid
    template_bank["gamma_grid_3D"] = gamma_grid
    template_bank["template_grid_3D"] = np.empty((nx_pts, ny_pts, nz_pts), dtype=object)

    # Create a list of all parameter combinations
    idx_list = list(np.ndindex(nx_pts, ny_pts, nz_pts))

    # Use Pool to parallelize the computation
    with Pool(cpu_count()) as pool:  # Use maximum number of cores
        results = pool.starmap(
            compute_RP_template,
            [(t_params, omega_grid, theta_grid, gamma_grid, idx) for idx in idx_list],
        )
    # Store the results in the templates dictionary
    for idx, template in results:
        template_bank["template_grid_3D"][idx] = template

    return template_bank


def create_RP_templates_mtx(t_params: dict):
    nx_pts = 41
    ny_pts = 151
    nz_pts = 101
    omega_arr = np.linspace(0, 4, nx_pts)
    theta_arr = np.linspace(0, 15, ny_pts)
    gamma_arr = np.linspace(0, 2 * np.pi, nz_pts)

    # Create a 3D meshgrid
    omega_grid, theta_grid, gamma_grid = np.meshgrid(
        omega_arr, theta_arr, gamma_arr, indexing="ij"
    )

    # Initialize an empty dict to store the templates
    template_grid = np.empty((nx_pts, ny_pts, nz_pts), dtype=object)

    # Create a list of all parameter combinations
    idx_list = list(np.ndindex(nx_pts, ny_pts, nz_pts))

    # Use Pool to parallelize the computation
    with Pool(cpu_count()) as pool:  # Use maximum number of cores
        results = pool.starmap(
            compute_RP_template,
            [(t_params, omega_grid, theta_grid, gamma_grid, idx) for idx in idx_list],
        )
    # Store the results in the templates dictionary
    for idx, template in results:
        template_grid[idx] = template

    np.savez_compressed(
        "data/template_bank.npz",
        template_grid=template_grid,
        omega_grid=omega_grid,
        theta_grid=theta_grid,
        gamma_grid=gamma_grid,
        omega_arr=omega_arr,
        theta_arr=theta_arr,
        gamma_arr=gamma_arr,
        t_params=t_params,
    )


def compute_mismatch(
    t_strain: FrequencySeries,
    s_strain: FrequencySeries,
    f_min=20,
    psd=None,
    use_opt_match=True,
) -> dict:
    # Get the psd from s_strain; Should provide psd to avoid recomputing it and save time
    if psd is None:
        f_arr = s_strain.sample_frequencies + f_min
        psd = Sn(f_arr)

    match_func = optimized_match if use_opt_match else match
    match_val, index, phi = match_func(t_strain, s_strain, psd, return_phase=True)  # type: ignore
    mismatch = 1 - match_val

    return {"mismatch": mismatch, "index": index, "phi": phi}


def create_mismatch_contour(
    template_bank: dict,
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

    # Extract template FrequencySeries objects from the template bank
    template_grid_3D = template_bank["template_grid_3D"]

    # Compute the mismatches in parallel
    with Pool(cpu_count()) as pool:
        results = pool.starmap(
            compute_mismatch,
            [
                (t_strain, s_strain, f_min, psd, use_opt_match)
                for t_strain in template_grid_3D.flatten()
            ],
        )

    # Initialize 3D grids
    omega_grid_3D = template_bank["omega_grid_3D"]
    ep_grid_3D = np.zeros_like(omega_grid_3D)
    idx_grid_3D = np.zeros_like(omega_grid_3D)
    phi_grid_3D = np.zeros_like(omega_grid_3D)

    # Populate 3D grids with mismatch results
    for i, result in enumerate(results):
        i_3D = np.unravel_index(i, template_grid_3D.shape)
        ep_grid_3D[i_3D] = result["mismatch"]
        idx_grid_3D[i_3D] = result["index"]
        phi_grid_3D[i_3D] = result["phi"]

    # Initialize 2D grids
    omega_arr = template_bank["omega_array"]
    theta_arr = template_bank["theta_array"]
    gamma_arr = template_bank["gamma_array"]
    omega_grid_2D, theta_grid_2D = np.meshgrid(omega_arr, theta_arr, indexing="ij")
    g_min_grid_2D = np.zeros_like(omega_grid_2D)
    ep_grid_2D = np.zeros_like(omega_grid_2D)

    # Find the gamma_P that minimizes the mismatch for each omega_tilde and theta_tilde pair
    for i, omega in enumerate(omega_arr):
        for j, theta in enumerate(theta_arr):
            gamma_min_idx = np.argmin(ep_grid_3D[i, j, :])
            g_min_grid_2D[i, j] = gamma_arr[gamma_min_idx]
            ep_grid_2D[i, j] = ep_grid_3D[i, j, gamma_min_idx]

    return {
        "omega_grid_2D": omega_grid_2D,
        "theta_grid_2D": theta_grid_2D,
        "epsilon_grid_2D": ep_grid_2D,
        "gamma_min_grid_2D": g_min_grid_2D,
        "source_params": s_params,
        "template_params": template_bank["template_params"],
    }


#####################################
# Section 3: Dictionary of Contours #
#####################################


def create_contours_td(
    template_bank: dict,
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

    td_arr = np.round(td_arr, 6)  # Round to 6 decimal places
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
                template_bank, s_params, f_min, delta_f, psd
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
    template_bank: dict,
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

    I_arr = np.round(I_arr, 6)  # Round to 6 decimal places
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
                template_bank, s_params, f_min, delta_f, psd
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
# Section 4: Super Contour #
############################


def create_super_contour(
    template_bank: dict,
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

    td_arr = np.round(td_arr, 6)  # Round to 6 decimal places
    I_arr = np.round(I_arr, 6)  # Round to 6 decimal places
    results = {}

    for td in td_arr:
        results[td] = create_contours_I(
            template_bank, s_params, td, I_arr, f_min, delta_f, psd, what_template
        )

    results["td_arr"] = td_arr
    results["I_arr"] = I_arr
    results["source_params"] = s_params  # for convenience
    results["template_params"] = template_bank["template_params"]  # for convenience
    return results
