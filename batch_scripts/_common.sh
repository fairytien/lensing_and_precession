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