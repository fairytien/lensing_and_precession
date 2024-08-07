import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.contours_ver3 import *


@timer_decorator
def main():
    print("Number of cores: ", cpu_count())

    # Assign parameters
    RP_params = set_to_location(loc_params["Taman"]["edgeon"], RP_params_1)[0]
    mcz = 60
    RP_params["mcz"] = mcz * solar_mass

    create_RP_templates_npz(
        RP_params, "data/sys2_template_bank_mcz" + str(mcz) + ".npz"
    )

    # filepath = pickle_data(results, "data", "sys2_template_bank_mcz" + str(mcz))


if __name__ == "__main__":
    main()
