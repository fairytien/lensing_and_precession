import os
from modules.contours_ver2 import *


@timer_decorator
def main():
    print("Number of cores: ", cpu_count())

    # Get the array index from the environment variable
    # idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))

    # Assign parameters
    lens_params, RP_params = set_to_location(
        loc_params["Taman"]["random"], lens_params_1, RP_params_1
    )
    mcz = 30
    lens_params["mcz"] = RP_params["mcz"] = mcz * solar_mass
    RP_params["omega_tilde"] = 3.0
    limits = get_lens_limits_for_RP_L(RP_params, lower=0.5)
    td_min, td_max = limits["td_min"], limits["td_max"]
    td_arr = np.linspace(td_min, td_max, 100)

    print("Finished assigning parameters")

    results = create_contours_td(RP_params, lens_params, 0.5, td_arr)

    filepath = pickle_data(results, "data", "TACC_contours_mcz" + str(mcz) + "_td")


if __name__ == "__main__":
    main()
