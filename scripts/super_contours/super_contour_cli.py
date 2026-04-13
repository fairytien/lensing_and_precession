import sys, os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from modules.legacy.contours_v2 import *

import argparse


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Create super contour with configurable system, mcz, td_arr, and I_arr."
    )

    parser.add_argument(
        "--system",
        type=str,
        default="sys2",
        choices=[
            "sys1",
            "sys2",
            "sys3",
            "1",
            "2",
            "3",
            "taman-faceon",
            "taman-edgeon",
            "taman-random",
        ],
        help=(
            "Which system configuration to use. "
            "sys1=Taman/faceon, sys2=Taman/edgeon, sys3=Taman/random."
        ),
    )

    parser.add_argument(
        "--mcz",
        type=float,
        default=20,
        help="Chirp mass (mcz) in solar masses.",
    )

    td_group = parser.add_mutually_exclusive_group()
    td_group.add_argument(
        "--td-list",
        type=float,
        nargs="+",
        help="Explicit list of td values (seconds). Overrides --td-start/--td-stop/--td-n.",
    )
    parser.add_argument(
        "--td-start",
        type=float,
        default=0.02,
        help="Start of td range (seconds) when generating linspace.",
    )
    parser.add_argument(
        "--td-stop",
        type=float,
        default=0.07,
        help="End of td range (seconds) when generating linspace.",
    )
    parser.add_argument(
        "--td-n",
        type=int,
        default=40,
        help="Number of td samples when generating linspace.",
    )

    I_group = parser.add_mutually_exclusive_group()
    I_group.add_argument(
        "--I-list",
        type=float,
        nargs="+",
        help="Explicit list of inclination values I. Overrides --I-start/--I-stop/--I-n.",
    )
    parser.add_argument(
        "--I-start",
        type=float,
        default=0.1,
        help="Start of I range when generating linspace.",
    )
    parser.add_argument(
        "--I-stop",
        type=float,
        default=0.9,
        help="End of I range when generating linspace.",
    )
    parser.add_argument(
        "--I-n",
        type=int,
        default=40,
        help="Number of I samples when generating linspace.",
    )

    parser.add_argument(
        "--nsplit",
        type=int,
        default=10,
        help="Number of splits for td array across SLURM array jobs.",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="If set, do not split td array by SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=None,
        help="Manually set split index (overrides SLURM_ARRAY_TASK_ID).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/super_contours",
        help="Output directory for pickle.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="TACC",
        help="Prefix for output filename.",
    )

    parser.add_argument(
        "--lores",
        action="store_true",
        help="Use low-resolution implementation (modules.legacy.contours_v2_lores).",
    )

    return parser.parse_args()


def _resolve_system(system_flag):
    flag = system_flag.lower()
    if flag in ("sys1", "1", "taman-faceon"):
        return 1, loc_params["Taman"]["faceon"]
    if flag in ("sys2", "2", "taman-edgeon"):
        return 2, loc_params["Taman"]["edgeon"]
    if flag in ("sys3", "3", "taman-random"):
        return 3, loc_params["Taman"]["random"]
    raise ValueError(f"Unknown system: {system_flag}")


@timer_decorator
def main():
    print("Number of cores: ", cpu_count())

    args = _parse_args()

    env_idx = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
    idx = args.idx if args.idx is not None else env_idx

    sys_id, location = _resolve_system(args.system)

    lens_params, RP_params = set_to_location(location, lens_params_1, RP_params_1)

    mcz = args.mcz
    lens_params["mcz"] = RP_params["mcz"] = mcz * solar_mass

    if args.td_list is not None:
        td_arr_long = np.array(args.td_list, dtype=float)
    else:
        td_arr_long = np.linspace(args.td_start, args.td_stop, args.td_n)

    if args.no_split:
        td_arr = td_arr_long
        idx_used = 0
    else:
        splits = np.array_split(td_arr_long, args.nsplit)
        if len(splits) == 0:
            raise ValueError("Empty td array after splitting.")
        idx_used = idx % len(splits)
        td_arr = splits[idx_used]

    if args.I_list is not None:
        I_arr = np.array(args.I_list, dtype=float)
    else:
        I_arr = np.linspace(args.I_start, args.I_stop, args.I_n)

    print("Finished assigning parameters")

    if args.lores:
        from modules.legacy.contours_v2_lores import (
            create_super_contour as create_super_contour_impl,
        )
    else:
        from modules.legacy.contours_v2 import (
            create_super_contour as create_super_contour_impl,
        )

    results = create_super_contour_impl(RP_params, lens_params, td_arr, I_arr)

    fast_tag = "_lores" if args.lores else ""
    filename = (
        f"{args.output_prefix}_sys{sys_id}{fast_tag}_super_contour_mcz"
        + str(mcz)
        + f"_{idx_used}"
    )
    filepath = pickle_data(results, args.output_dir, filename)
    return filepath


if __name__ == "__main__":
    main()
