import os
from modules.Classes_ver2 import *
from modules.default_params_ver1 import *
from modules.functions_ver2 import *
from modules.contours_ver1 import *


@timer_decorator
def main():
    print("Number of cores: ", cpu_count())

    # Get the array index from the environment variable
    # idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))

    mcz = 30
    lens_params_1["mcz"] = RP_params_1["mcz"] = mcz * solar_mass
    lens_params_1["y"] = 0.3
    lens_params, RP_params = set_to_params(lens_params_1, RP_params_1)

    # Assign MLz limits and parameter arrays
    RP_params["omega_tilde"] = 3.0
    limits = get_lens_limits_for_RP_L(RP_params, lower=0.5)
    MLz_min, MLz_max = limits["MLz_min"], limits["MLz_max"]
    MLz_arr_long = np.linspace(MLz_min, MLz_max, 200)
    MLz_arr = MLz_arr_long

    print("Finished assigning parameters")

    results = create_mismatch_contours_td(RP_params, lens_params, MLz_arr)

    filepath = pickle_data(results, "data", "TACC_contours_mcz30_td")


if __name__ == "__main__":
    main()
