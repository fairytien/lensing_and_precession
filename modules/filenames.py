import os


def bank_filename(
    bank_dir: str,
    mcz_msun: float,
    omega_min: float,
    omega_max: float,
    omega_pts: int,
    theta_min: float,
    theta_max: float,
    theta_pts: int,
    gamma_pts: int,
    f_min: float,
    delta_f: float,
    orientation_tag: str,
    prefix: str = "rp_bank",
) -> str:
    os.makedirs(bank_dir, exist_ok=True)
    name = (
        f"{prefix}_mcz{mcz_msun:.0f}_omega{omega_min:.0f}-{omega_max:.0f}"
        f"_theta{theta_min:.0f}-{theta_max:.0f}_o{omega_pts}xt{theta_pts}xg{gamma_pts}"
        f"_f{int(f_min)}_df{delta_f:.2f}_{orientation_tag}.h5"
    )
    return os.path.join(bank_dir, name)


def mismatch_cubes_filename(
    results_dir: str,
    mcz_msun: float,
    td_min_ms: float,
    td_max_ms: float,
    f_min: float,
    delta_f: float,
    orientation_tag: str,
) -> str:
    os.makedirs(results_dir, exist_ok=True)
    name = (
        f"mismatch_cubes_mcz{mcz_msun:.0f}Msun_td{td_min_ms:.0f}-{td_max_ms:.0f}ms"
        f"_f{int(f_min)}_df{delta_f:.2f}_{orientation_tag}.h5"
    )
    return os.path.join(results_dir, name)


def best_match_filename(
    results_dir: str,
    td_min_ms: float,
    td_max_ms: float,
    mcz_min: float,
    mcz_max: float,
    f_min: float,
    delta_f: float,
    orientation_tag: str,
) -> str:
    os.makedirs(results_dir, exist_ok=True)
    name = (
        f"best_match_td{td_min_ms:.0f}-{td_max_ms:.0f}ms"
        f"_mcz{mcz_min:.0f}-{mcz_max:.0f}Msun_f{int(f_min)}_df{delta_f:.2f}_{orientation_tag}.h5"
    )
    return os.path.join(results_dir, name)


def contour_td_mcz_filename(
    fig_dir: str,
    td_min_ms: float,
    td_max_ms: float,
    mcz_min: float,
    mcz_max: float,
    orientation_tag: str,
    ext: str = "pdf",
) -> str:
    os.makedirs(fig_dir, exist_ok=True)
    name = (
        f"contour_td{td_min_ms:.0f}-{td_max_ms:.0f}ms_"
        f"mcz{mcz_min:.0f}-{mcz_max:.0f}Msun_min_mismatch_{orientation_tag}.{ext}"
    )
    return os.path.join(fig_dir, name)
