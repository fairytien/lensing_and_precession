import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.contours_ver2 import *


@timer_decorator
def main():
    print("Number of cores: ", cpu_count())

    # Get the array index from the environment variable
    idx = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))

    # Assign parameters
    lens_params, RP_params = set_to_location(
        loc_params["Taman"]["random"], lens_params_1, RP_params_1
    )
    mcz = 20
    lens_params["mcz"] = RP_params["mcz"] = mcz * solar_mass
    RP_params["omega_tilde"] = 3
    limits = get_lens_limits_for_RP_L(RP_params)
    td_min, td_max = limits["td_min"], limits["td_max"]
    td_arr_long = np.linspace(td_min, td_max, 40)
    td_arr = np.array_split(td_arr_long, 4)[idx]
    I_arr = np.linspace(0.1, 0.9, 40)

    print("Finished assigning parameters")

    results = get_super_contour(RP_params, lens_params, td_arr, I_arr)

    filepath = pickle_data(
        results, "data", "TACC_super_contour_mcz" + str(mcz) + "_" + str(idx)
    )


if __name__ == "__main__":
    main()
