#############################
# Section 1: Import Modules #
#############################


# import py scripts
from modules.functions_v3 import *

# import libraries
from multiprocessing import Pool, cpu_count


###############################
# Section 2: RP Template Bank #
###############################


def compute_RP_template(
    tpl_params: dict,
    omega_grid: np.ndarray,
    theta_grid: np.ndarray,
    gamma_grid: np.ndarray,
    idx: tuple,
    **kwargs,
) -> tuple[tuple, np.ndarray]:

    # Compute the template based on t_params
    tpl_params_copy = copy.deepcopy(tpl_params)
    tpl_params_copy["omega_tilde"] = omega_grid[idx]
    tpl_params_copy["theta_tilde"] = theta_grid[idx]
    tpl_params_copy["gamma_P"] = gamma_grid[idx]

    template = get_gw(tpl_params_copy, frequencySeries=False, **kwargs)["strain"]

    return idx, template


def create_RP_templates(
    tpl_params: dict, n_omega: int, n_theta: int, n_gamma: int, filename: str, npz=True
) -> np.ndarray:
    omega_arr = np.linspace(0, 6, n_omega)
    theta_arr = np.linspace(0, 15, n_theta)
    gamma_arr = np.linspace(0, 2 * np.pi, n_gamma, endpoint=False)

    # Create a 3D meshgrid
    omega_grid, theta_grid, gamma_grid = np.meshgrid(
        omega_arr, theta_arr, gamma_arr, indexing="ij"
    )

    # Initialize an empty grid to store the templates
    template_grid = np.empty((n_omega, n_theta, n_gamma), dtype=object)

    # Create a list of indices to parallelize the computation
    idx_list = list(np.ndindex(n_omega, n_theta, n_gamma))

    # Use Pool to parallelize the computation
    with Pool(cpu_count()) as pool:  # Use maximum number of cores
        results = pool.starmap(
            compute_RP_template,
            [(tpl_params, omega_grid, theta_grid, gamma_grid, idx) for idx in idx_list],
        )
    # Store the results in the template grid
    for idx, template in results:
        template_grid[idx] = template

    if npz:
        np.savez_compressed(
            filename, template_grid=template_grid, template_params=tpl_params
        )

    return template_grid


#################################
# Section 3: A Mismatch Contour #
#################################


def compute_mismatch(
    t_strain: Union[np.ndarray, FrequencySeries],
    s_strain: Union[np.ndarray, FrequencySeries],
    f_min: float = 20,
    delta_f: float = 0.25,
    psd: FrequencySeries = None,
    use_opt_match=True,
) -> dict:
    if not isinstance(t_strain, FrequencySeries):
        t_strain = FrequencySeries(t_strain, delta_f)
    if not isinstance(s_strain, FrequencySeries):
        s_strain = FrequencySeries(s_strain, delta_f)

    # Get the psd from s_strain; Should provide psd to avoid recomputing it and save time
    if psd is None:
        f_arr = s_strain.sample_frequencies + f_min
        psd = Sn(f_arr, f_min=f_min, delta_f=delta_f)

    match_func = optimized_match if use_opt_match else match
    match_val, index, phi = match_func(t_strain, s_strain, psd, return_phase=True)  # type: ignore
    mismatch = 1 - match_val

    return {"mismatch": mismatch, "index": index, "phi": phi}


def create_mismatch_contour(
    template_grid: np.ndarray,
    s_params: dict,
    f_min: float = 20,
    delta_f: float = 0.25,
    psd: FrequencySeries = None,
    use_opt_match=True,
) -> dict:
    # Get the strain from the source parameters
    s_strain = get_gw(s_params, f_min, delta_f)["strain"]

    # Get the psd from s_strain; Should provide psd to avoid recomputing it and save time
    if psd is None:
        f_arr = s_strain.sample_frequencies + f_min
        psd = Sn(f_arr, f_min=f_min, delta_f=delta_f)

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
