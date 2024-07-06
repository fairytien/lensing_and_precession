import os
from modules.Classes_ver2 import *
from modules.default_params_ver2 import *
from modules.functions_ver2 import *
from modules.contours_ver2 import *


@timer_decorator
def main():
    print("Number of cores: ", cpu_count())

    # Get the array index from the environment variable
    # idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))

    lens_params, RP_params = set_to_location(
        loc_params["Taman"]["random"], lens_params_1, RP_params_1
    )
    mcz = 30
    lens_params["mcz"] = RP_params["mcz"] = mcz * solar_mass

    # Assign parameters
    I_arr = np.linspace(0.1, 0.5, 100)

    print("Finished assigning parameters")

    results = create_mismatch_contours_I(RP_params, lens_params, 0.02, I_arr)

    filepath = pickle_data(results, "data", "TACC_contours_mcz" + str(mcz) + "_I")


if __name__ == "__main__":
    main()
