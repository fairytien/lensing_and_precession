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
        loc_params["Taman"]["edgeon"], lens_params_1, RP_params_1
    )
    mcz = 30
    lens_params["mcz"] = RP_params["mcz"] = mcz * solar_mass
    td_arr_long = np.linspace(0.02, 0.07, 40)  # To be in geometric optics regime
    td_arr = np.array_split(td_arr_long, 10)[idx]
    I_arr = np.linspace(0.1, 0.9, 40)
    print("Finished assigning parameters")

    results = create_super_contour(RP_params, lens_params, td_arr, I_arr)

    filepath = pickle_data(
        results, "data", "TACC_sys2_super_contour_mcz" + str(mcz) + "_" + str(idx)
    )


if __name__ == "__main__":
    main()
