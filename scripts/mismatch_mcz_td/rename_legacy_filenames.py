"""Rename legacy mismatch-pipeline filenames to the canonical naming convention.

This utility is intentionally one-way:
- Reads known legacy filename patterns.
- Computes canonical targets via modules.filenames builders.
- Renames files in place.

By default it performs a dry run; use --apply to execute renames.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import h5py

from modules.filenames import (
    bank_filename,
    best_match_mcz_td_filename,
    contour_mcz_td_filename,
    mismatch_cube_filename,
)


def _to_float(token: str) -> float:
    return float(str(token).replace("p", "."))


_NUM = r"[0-9]+(?:[p\.][0-9]+)?(?:[eE][+\-]?[0-9]+)?"


@dataclass
class RenamePlan:
    source: str
    target: str


def _iter_files(directory: str, suffixes: Tuple[str, ...]) -> Iterable[str]:
    if not os.path.isdir(directory):
        return
    for name in os.listdir(directory):
        if name.endswith(suffixes):
            yield os.path.join(directory, name)


def _plan_bank_renames(bank_dir: str) -> List[RenamePlan]:
    # legacy: {prefix}_mcz{mcz}Msun_omega{min}-{max}_theta{min}-{max}_o{o}-t{t}-g{g}_{tag}.h5
    pat = re.compile(
        rf"^(?P<prefix>.+?)(?:_z(?P<z>{_NUM}))?_mcz(?P<mcz>{_NUM})Msun"
        rf"_omega(?P<omin>{_NUM})-(?P<omax>{_NUM})"
        rf"_theta(?P<tmin>{_NUM})-(?P<tmax>{_NUM})"
        r"_o(?P<op>\d+)-t(?P<tp>\d+)-g(?P<gp>\d+)_(?P<tag>.+)\.h5$"
    )
    plans: List[RenamePlan] = []
    for path in _iter_files(bank_dir, (".h5",)) or []:
        m = pat.match(os.path.basename(path))
        if not m:
            continue
        g = m.groupdict()
        z = _to_float(g["z"]) if g["z"] is not None else None
        target = bank_filename(
            bank_dir=bank_dir,
            mcz_msun=_to_float(g["mcz"]),
            omega_min=_to_float(g["omin"]),
            omega_max=_to_float(g["omax"]),
            omega_pts=int(g["op"]),
            theta_min=_to_float(g["tmin"]),
            theta_max=_to_float(g["tmax"]),
            theta_pts=int(g["tp"]),
            gamma_pts=int(g["gp"]),
            orientation_tag=g["tag"],
            z=z,
            prefix=g["prefix"],
        )
        if os.path.abspath(path) != os.path.abspath(target):
            plans.append(RenamePlan(source=path, target=target))
    return plans


def _plan_mismatch_cube_renames(results_dir: str) -> List[RenamePlan]:
    mismatch_dir = os.path.join(results_dir, "mismatch_cubes")
    # legacy: mismatch_cubes_mcz{mcz}Msun_I{I}_td{min}-{max}ms_td{td}-o{o}-t{t}-g{g}_{tag}.h5
    pat = re.compile(
        rf"^mismatch_cubes(?:_z(?P<z>{_NUM}))?_mcz(?P<mcz>{_NUM})Msun"
        rf"_I(?P<I>{_NUM})_td(?P<tdmin>{_NUM})-(?P<tdmax>{_NUM})ms"
        r"_td(?P<tdp>\d+)-o(?P<op>\d+)-t(?P<tp>\d+)-g(?P<gp>\d+)_(?P<tag>.+)\.h5$"
    )
    plans: List[RenamePlan] = []
    for path in _iter_files(mismatch_dir, (".h5",)) or []:
        m = pat.match(os.path.basename(path))
        if not m:
            continue
        g = m.groupdict()
        z = _to_float(g["z"]) if g["z"] is not None else None
        try:
            with h5py.File(path, "r") as h5:
                omega = h5["omega"][:]
                theta = h5["theta"][:]
            omega_min = float(min(omega))
            omega_max = float(max(omega))
            theta_min = float(min(theta))
            theta_max = float(max(theta))
        except Exception:
            continue
        target = mismatch_cube_filename(
            results_dir=results_dir,
            mcz_msun=_to_float(g["mcz"]),
            I=_to_float(g["I"]),
            td_min_ms=_to_float(g["tdmin"]),
            td_max_ms=_to_float(g["tdmax"]),
            td_pts=int(g["tdp"]),
            omega_min=omega_min,
            omega_max=omega_max,
            omega_pts=int(g["op"]),
            theta_min=theta_min,
            theta_max=theta_max,
            theta_pts=int(g["tp"]),
            gamma_pts=int(g["gp"]),
            orientation_tag=g["tag"],
            z=z,
        )
        if os.path.abspath(path) != os.path.abspath(target):
            plans.append(RenamePlan(source=path, target=target))
    return plans


def _plan_best_match_renames(results_dir: str) -> List[RenamePlan]:
    best_dir = os.path.join(results_dir, "best_match")
    # legacy: best_match_I{I}_mcz{min}-{max}Msun_td{min}-{max}ms_m{m}-td{td}-o{o}-t{t}-g{g}_{tag}.h5
    pat = re.compile(
        rf"^best_match_I(?P<I>{_NUM})(?:_z(?P<z>{_NUM}))?"
        rf"_mcz(?P<mmin>{_NUM})-(?P<mmax>{_NUM})Msun"
        rf"_td(?P<tdmin>{_NUM})-(?P<tdmax>{_NUM})ms"
        r"_m(?P<mp>\d+)-td(?P<tdp>\d+)-o(?P<op>\d+)-t(?P<tp>\d+)-g(?P<gp>\d+)"
        r"_(?P<tag>.+)\.h5$"
    )
    plans: List[RenamePlan] = []
    for path in _iter_files(best_dir, (".h5",)) or []:
        m = pat.match(os.path.basename(path))
        if not m:
            continue
        g = m.groupdict()
        z = _to_float(g["z"]) if g["z"] is not None else None
        omega_min = omega_max = None
        theta_min = theta_max = None
        try:
            with h5py.File(path, "r") as h5:
                attrs = h5.attrs
                if (
                    "template_param_omega_min" in attrs
                    and "template_param_omega_max" in attrs
                ):
                    omega_min = float(attrs["template_param_omega_min"])
                    omega_max = float(attrs["template_param_omega_max"])
                if (
                    "template_param_theta_min" in attrs
                    and "template_param_theta_max" in attrs
                ):
                    theta_min = float(attrs["template_param_theta_min"])
                    theta_max = float(attrs["template_param_theta_max"])
        except Exception:
            pass
        target = best_match_mcz_td_filename(
            results_dir=results_dir,
            I=_to_float(g["I"]),
            mcz_min=_to_float(g["mmin"]),
            mcz_max=_to_float(g["mmax"]),
            mcz_pts=int(g["mp"]),
            td_min_ms=_to_float(g["tdmin"]),
            td_max_ms=_to_float(g["tdmax"]),
            td_pts=int(g["tdp"]),
            omega_min=omega_min,
            omega_max=omega_max,
            omega_pts=int(g["op"]),
            theta_min=theta_min,
            theta_max=theta_max,
            theta_pts=int(g["tp"]),
            gamma_pts=int(g["gp"]),
            orientation_tag=g["tag"],
            z=z,
        )
        if os.path.abspath(path) != os.path.abspath(target):
            plans.append(RenamePlan(source=path, target=target))
    return plans


def _plan_contour_renames(fig_dir: str) -> List[RenamePlan]:
    # legacy: contour_I{I}_mcz{min}-{max}Msun_td{min}-{max}ms_min_mismatch_{tag}.pdf
    pat = re.compile(
        rf"^contour_I(?P<I>{_NUM})(?:_z(?P<z>{_NUM}))?"
        rf"_mcz(?P<mmin>{_NUM})-(?P<mmax>{_NUM})Msun"
        rf"_td(?P<tdmin>{_NUM})-(?P<tdmax>{_NUM})ms"
        r"_min_mismatch_(?P<tag>.+)\.(?P<ext>pdf|png)$"
    )
    plans: List[RenamePlan] = []
    for path in _iter_files(fig_dir, (".pdf", ".png")) or []:
        m = pat.match(os.path.basename(path))
        if not m:
            continue
        g = m.groupdict()
        z = _to_float(g["z"]) if g["z"] is not None else None
        target = contour_mcz_td_filename(
            fig_dir=fig_dir,
            I=_to_float(g["I"]),
            mcz_min=_to_float(g["mmin"]),
            mcz_max=_to_float(g["mmax"]),
            mcz_pts=None,
            td_min_ms=_to_float(g["tdmin"]),
            td_max_ms=_to_float(g["tdmax"]),
            td_pts=None,
            orientation_tag=g["tag"],
            z=z,
            ext=g["ext"],
        )
        if os.path.abspath(path) != os.path.abspath(target):
            plans.append(RenamePlan(source=path, target=target))
    return plans


def _dedupe_by_target(plans: List[RenamePlan]) -> List[RenamePlan]:
    seen: Dict[str, RenamePlan] = {}
    for p in plans:
        seen[p.target] = p
    return list(seen.values())


def _apply_renames(plans: List[RenamePlan], apply: bool) -> None:
    if not plans:
        print("No legacy files matched rename patterns.")
        return

    conflicts = [p for p in plans if os.path.exists(p.target)]
    if conflicts:
        print("Conflicts detected (target exists); skipping these:")
        for p in conflicts:
            print(f"  SKIP {p.source} -> {p.target}")
        plans = [p for p in plans if p not in conflicts]

    if not plans:
        print("No renames to perform after conflict checks.")
        return

    mode = "APPLY" if apply else "DRY-RUN"
    for p in plans:
        print(f"[{mode}] {p.source} -> {p.target}")

    if not apply:
        print(f"Planned {len(plans)} rename(s). Use --apply to execute.")
        return

    for p in plans:
        os.rename(p.source, p.target)
    print(f"Renamed {len(plans)} file(s).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename legacy mismatch-pipeline artifacts to canonical naming."
    )
    parser.add_argument(
        "--bank_dir",
        default="data/template_banks",
        help="Directory containing template bank files.",
    )
    parser.add_argument(
        "--results_dir",
        default="data/contours_td_mcz",
        help="Root results directory containing mismatch_cubes/ and best_match/.",
    )
    parser.add_argument(
        "--figure_dir",
        default="figures/mismatch_mcz_td",
        help="Directory containing contour figure outputs.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute renames (default is dry run).",
    )
    args = parser.parse_args()

    plans: List[RenamePlan] = []
    plans.extend(_plan_bank_renames(args.bank_dir))
    plans.extend(_plan_mismatch_cube_renames(args.results_dir))
    plans.extend(_plan_best_match_renames(args.results_dir))
    plans.extend(_plan_contour_renames(args.figure_dir))

    plans = _dedupe_by_target(plans)
    plans.sort(key=lambda p: p.source)
    _apply_renames(plans, apply=args.apply)


if __name__ == "__main__":
    main()
