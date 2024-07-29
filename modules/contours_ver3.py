#############################
# Section 1: Import Modules #
#############################


# import py scripts
from modules.functions_ver2 import *

# import libraries
from multiprocessing import Pool, cpu_count
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator


################################################
# Section 2: Speed-Optimized Contour Functions #
################################################


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
    templates = {}
    templates["template_params"] = t_params
    templates["omega_array"] = omega_arr
    templates["theta_array"] = theta_arr
    templates["gamma_array"] = gamma_arr
    templates["omega_grid_3D"] = omega_grid
    templates["theta_grid_3D"] = theta_grid
    templates["gamma_grid_3D"] = gamma_grid

    def compute_RP_template(indices):
        i, j, k = indices
        omega = np.round(omega_grid[i, j, k], 6)
        theta = np.round(theta_grid[i, j, k], 6)
        gamma = np.round(gamma_grid[i, j, k], 6)

        coords = {"omega_tilde": omega, "theta_tilde": theta, "gamma_P": gamma}

        # Compute the template based on t_params
        t_params_copy = copy.deepcopy(t_params)
        t_params_copy["omega_tilde"] = omega
        t_params_copy["theta_tilde"] = theta
        t_params_copy["gamma_P"] = gamma

        template = get_gw(t_params_copy)["strain"]

        return coords, template

    # Create a list of all parameter combinations
    indices_list = [
        (i, j, k) for i in range(nx_pts) for j in range(ny_pts) for k in range(nz_pts)
    ]

    # Use Pool to parallelize the computation
    with Pool(cpu_count()) as pool:  # Use maximum number of cores
        results = pool.map(compute_RP_template, indices_list)

    # Store the results in the templates dictionary
    for coords, template in results:
        templates[coords] = template

    return templates


def compute_mismatch(
    t_strain: FrequencySeries,
    s_strain: FrequencySeries,
    f_min=20,
    psd=None,
    use_opt_match=True,
) -> dict:
    # Should provide psd to avoid recomputing it and speed up the computation
    if psd is None:
        f_arr = s_strain.sample_frequencies + f_min
        psd = Sn(f_arr)

    match_func = optimized_match if use_opt_match else match
    match_val, index, phi = match_func(t_strain, s_strain, psd, return_phase=True)  # type: ignore
    mismatch = 1 - match_val

    return {"mismatch": mismatch, "index": index, "phi": phi}


def create_mismatch_contour(
    templates: dict, s_strain: FrequencySeries, s_params: dict, f_min=20
) -> dict:
    # Get the psd from s_vec
    f_arr = s_strain.sample_frequencies + f_min
    psd = Sn(f_arr)

    # Extract template coordinates and FrequencySeries objects
    templates_coords = [
        coords for coords in templates.keys() if isinstance(coords, dict)
    ]
    templates_list = [templates[coords] for coords in templates_coords]

    # Compute the mismatches in parallel
    with Pool(cpu_count()) as pool:
        results = pool.starmap(
            compute_mismatch,
            [(t_strain, s_strain, f_min, psd) for t_strain in templates_list],
        )

    # Initialize 3D grids
    omega_grid_3D = templates["omega_grid_3D"]
    ep_grid_3D = np.zeros_like(omega_grid_3D)
    idx_grid_3D = np.zeros_like(omega_grid_3D)
    phi_grid_3D = np.zeros_like(omega_grid_3D)

    # Populate 3D grids with mismatch results
    for i, coords in enumerate(templates_coords):
        indices_3D = np.where(
            (templates["omega_grid_3D"] == coords["omega_tilde"])
            & (templates["theta_grid_3D"] == coords["theta_tilde"])
            & (templates["gamma_grid_3D"] == coords["gamma_P"])
        )
        ep_grid_3D[indices_3D] = results[i]["mismatch"]
        idx_grid_3D[indices_3D] = results[i]["index"]
        phi_grid_3D[indices_3D] = results[i]["phi"]

    # Initialize 2D grids
    omega_arr = templates["omega_array"]
    theta_arr = templates["theta_array"]
    gamma_arr = templates["gamma_array"]
    omega_grid_2D, theta_grid_2D = np.meshgrid(omega_arr, theta_arr, indexing="ij")
    gamma_min_grid_2D = np.zeros_like(omega_grid_2D)
    ep_grid_2D = np.zeros_like(omega_grid_2D)

    # Find the gamma_P that minimizes the mismatch for each omega_tilde and theta_tilde pair
    for i in range(len(omega_arr)):
        for j in range(len(theta_arr)):
            gamma_min_idx = np.argmin(ep_grid_3D[i, j, :])
            gamma_min_grid_2D[i, j] = gamma_arr[gamma_min_idx]
            ep_grid_2D[i, j] = ep_grid_3D[i, j, gamma_min_idx]

    return {
        "omega_grid_2D": omega_grid_2D,
        "theta_grid_2D": theta_grid_2D,
        "epsilon_grid_2D": ep_grid_2D,
        "gamma_min_grid_2D": gamma_min_grid_2D,
        "source_params": s_params,
        "template_params": templates["template_params"],
    }


################################################
# Section 2: Speed-Optimized Contour Functions #
################################################
