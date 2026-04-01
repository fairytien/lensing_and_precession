import sys, os, argparse

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from modules.contours_v3 import *


@timer_decorator
def main(output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print("Number of cores: ", cpu_count())

    # Assign parameters
    RP_params = set_orientation(orient_params["Taman"]["edgeon"], RP_params_1)[0]
    mcz_msun = 40
    RP_params["mcz"] = mcz_msun * SOLMASS2SEC

    create_RP_templates(
        RP_params,
        output_dir + "/sys2_template_grid_mcz" + str(mcz_msun) + ".npz",
        npz=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save data to a specified directory.")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save output data."
    )
    args = parser.parse_args()
    main(args.output_dir)
