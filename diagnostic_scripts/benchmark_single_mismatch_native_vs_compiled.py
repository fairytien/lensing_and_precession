#!/usr/bin/env python3
"""Benchmark one mismatch calculation: native Python vs compiled Cython module.

This script compares a single mismatch calculation in two modes:
- compiled: regular import of modules.match_utils (prefers .so extension)
- python: explicit source-load of modules/match_utils.py (bypasses extension import)

By default it benchmarks mismatch_from_params on one fixed parameter set.
It can also benchmark mismatch_from_strains on synthetic FrequencySeries inputs.
"""

from __future__ import annotations

import argparse
import cProfile
import importlib
import importlib.util
import json
import pstats
import subprocess
import sys
import time
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any, Dict

import numpy as np
from pycbc.types import FrequencySeries

ROOT = Path(__file__).resolve().parents[1]
MODULES_DIR = ROOT / "modules"
MATCH_UTILS_PY = MODULES_DIR / "match_utils.py"


def _is_compiled_module(module: Any) -> bool:
    cy_mod = getattr(module, "cython", None)
    return bool(getattr(cy_mod, "compiled", False))


def _load_match_utils(mode: str):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    if mode == "compiled":
        module = importlib.import_module("modules.match_utils")
        return module

    if mode == "python":
        spec = importlib.util.spec_from_file_location(
            "bench_match_utils_python", MATCH_UTILS_PY
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not create import spec for {MATCH_UTILS_PY}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    raise ValueError(f"Unknown mode: {mode}")


def _build_test_case(n_freq: int, delta_f: float, seed: int):
    rng = np.random.default_rng(seed)

    t_real = rng.normal(size=n_freq)
    t_imag = rng.normal(size=n_freq)
    s_real = t_real + 0.05 * rng.normal(size=n_freq)
    s_imag = t_imag + 0.05 * rng.normal(size=n_freq)

    t_strain = FrequencySeries(t_real + 1j * t_imag, delta_f=delta_f)
    s_strain = FrequencySeries(s_real + 1j * s_imag, delta_f=delta_f)

    psd_arr = np.linspace(1.0, 2.0, n_freq, dtype=np.float64)
    psd = FrequencySeries(psd_arr, delta_f=delta_f)

    return t_strain, s_strain, psd


def _build_param_test_case():
    from modules.default_params import RP_params_1, lens_params_1

    s_params = deepcopy(lens_params_1)
    t_params = deepcopy(RP_params_1)

    # Fix one representative lens/precession pair for a stable, single-call benchmark.
    s_params["y"] = 0.25
    s_params["MLz"] = 2e3 * s_params["mcz"]
    t_params["theta_tilde"] = 4.0
    t_params["omega_tilde"] = 2.0
    t_params["gamma_P"] = 0.0
    return t_params, s_params


def _profile_block(enabled: bool, repeats: int, fn):
    if not enabled:
        return fn(), None

    profiler = cProfile.Profile()
    profiler.enable()
    out = fn()
    profiler.disable()

    stats_buffer = StringIO()
    pstats.Stats(profiler, stream=stats_buffer).sort_stats("cumtime").print_stats(20)
    profile_text = stats_buffer.getvalue()
    return out, {
        "repeats_profiled": int(repeats),
        "top20_cumtime": profile_text,
    }


def _run_benchmark(
    mode: str,
    benchmark_kind: str,
    repeats: int,
    warmup: int,
    n_freq: int,
    delta_f: float,
    seed: int,
    use_opt_match: bool,
    profile: bool,
) -> Dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    module = _load_match_utils(mode)
    if benchmark_kind == "strains":
        mismatch_from_strains = module.mismatch_from_strains
        t_strain, s_strain, psd = _build_test_case(
            n_freq=n_freq, delta_f=delta_f, seed=seed
        )

        def call_once():
            return mismatch_from_strains(
                t_strain,
                s_strain,
                f_min=20.0,
                delta_f=delta_f,
                psd=psd,
                use_opt_match=use_opt_match,
                compare_both=False,
            )

    elif benchmark_kind == "params":
        mismatch_from_params = module.mismatch_from_params
        t_params, s_params = _build_param_test_case()

        def call_once():
            return mismatch_from_params(
                t_params,
                s_params,
                f_min=20.0,
                delta_f=delta_f,
                psd=None,
                use_opt_match=use_opt_match,
                compare_both=False,
            )

    else:
        raise ValueError(f"Unknown benchmark_kind: {benchmark_kind}")

    last_result = None
    for _ in range(max(0, warmup)):
        last_result = call_once()

    def _timed_loop():
        nonlocal last_result
        start = time.perf_counter()
        for _ in range(repeats):
            last_result = call_once()
        elapsed_local = time.perf_counter() - start
        return elapsed_local

    elapsed, profile_data = _profile_block(profile, repeats, _timed_loop)
    if last_result is None:
        raise RuntimeError("Benchmark did not produce a mismatch result")

    out = {
        "mode": mode,
        "benchmark_kind": benchmark_kind,
        "module_file": str(getattr(module, "__file__", "")),
        "module_compiled": _is_compiled_module(module),
        "repeats": int(repeats),
        "warmup": int(warmup),
        "n_freq": int(n_freq),
        "delta_f": float(delta_f),
        "seed": int(seed),
        "use_opt_match": bool(use_opt_match),
        "total_sec": float(elapsed),
        "per_call_sec": float(elapsed / repeats),
        "mismatch": float(last_result["mismatch"]),
        "index": float(last_result["index"]),
        "phi": float(last_result["phi"]),
    }
    if profile_data is not None:
        out["profile"] = profile_data
    return out


def _run_compare(args: argparse.Namespace):
    script_path = Path(__file__).resolve()

    def _run_child(mode: str) -> Dict[str, Any]:
        cmd = [
            sys.executable,
            str(script_path),
            "--mode",
            mode,
            "--benchmark-kind",
            args.benchmark_kind,
            "--repeats",
            str(args.repeats),
            "--warmup",
            str(args.warmup),
            "--n-freq",
            str(args.n_freq),
            "--delta-f",
            str(args.delta_f),
            "--seed",
            str(args.seed),
            "--json",
        ]
        if not args.use_opt_match:
            cmd.append("--no-opt-match")
        if args.profile:
            cmd.append("--profile")

        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            check=True,
            capture_output=True,
            text=True,
        )

        stdout = proc.stdout.strip().splitlines()
        if not stdout:
            raise RuntimeError(f"No output from child mode={mode}")
        return json.loads(stdout[-1])

    compiled = _run_child("compiled")
    python_native = _run_child("python")

    speedup = python_native["per_call_sec"] / compiled["per_call_sec"]
    summary = {
        "compiled": compiled,
        "python": python_native,
        "speedup_compiled_vs_python": float(speedup),
    }

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print("Single mismatch benchmark (native python vs compiled)")
    print(f"call kind: {args.benchmark_kind}")
    print(f"compiled module file: {compiled['module_file']}")
    print(f"python module file:   {python_native['module_file']}")
    print(
        f"compiled per call: {compiled['per_call_sec']:.6f} s | "
        f"python per call: {python_native['per_call_sec']:.6f} s"
    )
    print(f"speedup (python / compiled): {speedup:.3f}x")
    print(
        f"mismatch consistency: compiled={compiled['mismatch']:.12g}, "
        f"python={python_native['mismatch']:.12g}"
    )

    if args.profile:
        print("\n--- Compiled profile (top 20 cumulative) ---")
        print(compiled.get("profile", {}).get("top20_cumtime", ""))
        print("--- Python profile (top 20 cumulative) ---")
        print(python_native.get("profile", {}).get("top20_cumtime", ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark single mismatch call in Python vs compiled module."
    )
    parser.add_argument(
        "--mode",
        choices=["compare", "compiled", "python"],
        default="compare",
        help="compare runs both modes in subprocesses; compiled/python runs one mode only.",
    )
    parser.add_argument(
        "--benchmark-kind",
        choices=["strains", "params"],
        default="params",
        help="Benchmark mismatch_from_strains or mismatch_from_params.",
    )
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--n-freq", type=int, default=8192)
    parser.add_argument("--delta-f", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use-opt-match",
        dest="use_opt_match",
        action="store_true",
        default=True,
        help="Use optimized_match_bounded path (default: enabled).",
    )
    parser.add_argument(
        "--no-opt-match",
        dest="use_opt_match",
        action="store_false",
        help="Use standard match instead of optimized_match_bounded.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect cProfile stats for the timed loop and include top cumulative functions.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "compare":
        _run_compare(args)
        return

    result = _run_benchmark(
        mode=args.mode,
        benchmark_kind=args.benchmark_kind,
        repeats=args.repeats,
        warmup=args.warmup,
        n_freq=args.n_freq,
        delta_f=args.delta_f,
        seed=args.seed,
        use_opt_match=args.use_opt_match,
        profile=args.profile,
    )

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
