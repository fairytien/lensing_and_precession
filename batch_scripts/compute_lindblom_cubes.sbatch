#!/bin/bash
#SBATCH -J compute_lindblom # Job name
#SBATCH --account=PHY24021 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks=1 # Single task (not MPI)
#SBATCH --cpus-per-task=128 # Use all CPUs on node
#SBATCH --mem=0 # Use all available memory on node
#SBATCH -t 00:20:00 # Walltime (20 minutes per cube)
#SBATCH -p normal # Queue name where job is submitted
#SBATCH --array=0-80 # Array job for mcz values 10-90 (81 total)
#SBATCH -o batch_outputs/compute_lindblom_%A_%a.out # Output file
#SBATCH --mail-type=FAIL # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=tvn6@cornell.edu

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/batch_scripts"
source "$SCRIPT_DIR/_common.sh"
source "$SCRIPT_DIR/_contour_mcz_td_config.sh"

# Map array index to mcz value from configured grid
MCZ_VAL=$(python - <<'PY'
import os
import sys
import numpy as np

idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
mcz_min = float(os.environ["MCZ_MIN"])
mcz_max = float(os.environ["MCZ_MAX"])
mcz_pts = int(os.environ["MCZ_PTS"])

if idx < 0 or idx >= mcz_pts:
    print(
        f"Error: SLURM_ARRAY_TASK_ID={idx} is outside configured mcz grid size {mcz_pts}",
        file=sys.stderr,
    )
    raise SystemExit(2)

mcz_grid = np.linspace(mcz_min, mcz_max, mcz_pts, dtype=float)
print(f"{mcz_grid[idx]:g}")
PY
) || exit 2

export MCZ_VAL

# Paths
BANK_PREFIX="${BANK_PREFIX:-rp_bank}"

BANK_DIR_RESOLVED=$(python - <<'PY'
import os

from modules.filenames import template_bank_run_dir

z = float(os.environ.get("Z", "0"))
print(template_bank_run_dir(os.environ["BANK_DIR"], z))
PY
) || {
  echo "Error: failed to resolve template bank directory"
  exit 2
}

CUBE_PATH=$(python - <<'PY'
import os

from modules.filenames import contour_run_dir, mismatch_cube_filename

z = float(os.environ.get("Z", "0"))
results_root = os.environ["RESULTS_DIR"]
run_dir = contour_run_dir(
    results_root,
  I=float(os.environ["FLUX_RATIO"]),
    mcz_min=float(os.environ["MCZ_MIN"]),
    mcz_max=float(os.environ["MCZ_MAX"]),
    td_min_ms=float(os.environ["TD_MIN_MS"]),
    td_max_ms=float(os.environ["TD_MAX_MS"]),
    z=z,
)

path = mismatch_cube_filename(
    results_dir=run_dir,
    mcz_msun=float(os.environ["MCZ_VAL"]),
    I=float(os.environ["FLUX_RATIO"]),
    td_min_ms=float(os.environ["TD_MIN_MS"]),
    td_max_ms=float(os.environ["TD_MAX_MS"]),
    td_pts=int(os.environ["TD_PTS"]),
    omega_min=float(os.environ["OMEGA_MIN"]),
    omega_max=float(os.environ["OMEGA_MAX"]),
    omega_pts=int(os.environ["OMEGA_PTS"]),
    theta_min=float(os.environ["THETA_MIN"]),
    theta_max=float(os.environ["THETA_MAX"]),
    theta_pts=int(os.environ["THETA_PTS"]),
    gamma_pts=int(os.environ["GAMMA_PTS"]),
    orientation_tag=os.environ["ORIENT_PRESET"],
    z=z,
)
print(path)
PY
) || {
  echo "Error: failed to resolve canonical mismatch cube path"
  exit 2
}

# Check if cube exists
if [ ! -f "$CUBE_PATH" ]; then
  echo "Error: Mismatch cube not found: $CUBE_PATH"
  exit 1
fi

# Check if Lindblom cube already exists with SNR data
LINDBLOM_PATH="${CUBE_PATH%.h5}_lindblom.h5"
if [ -f "$LINDBLOM_PATH" ]; then
  # Check if it has SNR data
  if python -c "import h5py; h5 = h5py.File('$LINDBLOM_PATH', 'r'); has_snr = 'snr_cube' in h5; h5.close(); exit(0 if has_snr else 1)" 2>/dev/null; then
    echo "Lindblom cube with SNR already exists for mcz=$MCZ_VAL, skipping..."
    exit 0
  fi
fi

echo "Processing mcz=$MCZ_VAL (array task ${SLURM_ARRAY_TASK_ID})..."
echo "Cube path: $CUBE_PATH"
echo "Using $NWORKERS workers"

# Run the computation
python -m scripts.lindblom.compute_lindblom_contours \
  --cube_path "$CUBE_PATH" \
  --bank_dir "$BANK_DIR_RESOLVED" \
  --bank_prefix "$BANK_PREFIX" \
  --f_min 20.0 \
  --delta_f 0.25 \
  --n_workers "$NWORKERS"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "Successfully computed Lindblom cube for mcz=$MCZ_VAL"
else
  echo "Error computing Lindblom cube for mcz=$MCZ_VAL (exit code: $EXIT_CODE)"
  exit $EXIT_CODE
fi

