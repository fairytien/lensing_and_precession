import sys, os, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.contours_ver3 import *


@timer_decorator
def main(output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print("Number of cores: ", cpu_count())

    # Assign parameters
    results = {}
    mcz_arr = np.round(np.linspace(10, 90, 81), 2)
    lens_params, RP_params = set_to_location(
        loc_params["Taman"]["edgeon"], lens_params_1, RP_params_1
    )
    I = 0.5
    td = 0.03
    y = get_y_from_I(I)
    MLz = get_MLz_from_td(td, y)
    lens_params["y"] = y
    lens_params["MLz"] = MLz * solar_mass

    for mcz in mcz_arr:
        print(f"{mcz:.3g}")  # For tracking progress

        # Create RP template grid
        RP_params["mcz"] = mcz * solar_mass
        template_grid = create_RP_templates(
            RP_params,
            output_dir + "/sys2_template_grid_mcz" + str(mcz) + ".npz",
            npz=True,
        )

        # Create mismatch contour for each mcz value
        lens_params["mcz"] = mcz * solar_mass
        results[mcz] = create_mismatch_contour(template_grid, lens_params)

    filepath = pickle_data(results, "data", "sys2_contours_mcz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save data to a specified directory.")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save output data."
    )
    args = parser.parse_args()
    main(args.output_dir)
