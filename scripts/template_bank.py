import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.contours_ver3 import *


@timer_decorator
def main():
    print("Number of cores: ", cpu_count())

    # Assign parameters
    RP_params = set_to_location(loc_params["Taman"]["edgeon"], RP_params_1)[0]
    mcz = 20
    RP_params["mcz"] = mcz * solar_mass

    results = create_RP_templates(
        RP_params, "sys2_template_grid_mcz" + str(mcz), npz=False
    )

    filepath = pickle_data(results, "data", "sys2_template_grid_mcz" + str(mcz))


if __name__ == "__main__":
    main()
