#!/bin/bash

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

if command -v conda >/dev/null 2>&1; then
  # Change the default 'fairytien_gw' to your conda environment name,
  # or set the CONDA_ENV_NAME environment variable to override it.
  ENV_NAME="${CONDA_ENV_NAME:-fairytien_gw}"
  conda activate "$ENV_NAME" 2>/dev/null || {
    [ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc" >/dev/null 2>&1
    conda activate "$ENV_NAME" 2>/dev/null || {
      echo "Error: failed to activate conda env '$ENV_NAME'." >&2
      exit 2
    }
  }
fi

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

NWORKERS="${NWORKERS:-${SLURM_CPUS_PER_TASK:-128}}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---------------------------------------------------------------------------
# Shell helpers for building mcz / td grid CLI args (step vs pts).
# Usage: build_mcz_grid_args / build_td_grid_args append to a variable.
# ---------------------------------------------------------------------------
build_mcz_grid_args() {
  # Outputs: --mcz_min ... --mcz_max ... [--mcz_pts ... | --mcz_step ...]
  echo "--mcz_min ${MCZ_MIN} --mcz_max ${MCZ_MAX}"
  if [ -n "${MCZ_STEP:-}" ]; then
    echo "--mcz_step ${MCZ_STEP}"
  else
    echo "--mcz_pts ${MCZ_PTS}"
  fi
}

build_td_grid_args() {
  echo "--td_min_ms ${TD_MIN_MS} --td_max_ms ${TD_MAX_MS}"
  if [ -n "${TD_STEP_MS:-}" ]; then
    echo "--td_step_ms ${TD_STEP_MS}"
  else
    echo "--td_pts ${TD_PTS}"
  fi
}

build_I_grid_args() {
  # Outputs: --I_min ... --I_max ... [--I_pts ... | --I_step ...]
  echo "--I_min ${I_MIN} --I_max ${I_MAX}"
  if [ -n "${I_STEP:-}" ]; then
    echo "--I_step ${I_STEP}"
  else
    echo "--I_pts ${I_PTS}"
  fi
}