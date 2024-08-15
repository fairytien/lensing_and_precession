import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.contours_ver2 import *


@timer_decorator
def main():
    print("Number of cores: ", cpu_count())

    # Assign parameters
    results = {}
    mcz_arr = np.linspace(10, 90, 161)
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
        lens_params["mcz"] = RP_params["mcz"] = mcz * solar_mass
        results[mcz] = create_mismatch_contour_parallel(RP_params, lens_params)

    filepath = pickle_data(results, "data", "sys2_contours_mcz")


if __name__ == "__main__":
    main()
