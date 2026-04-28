#!/bin/bash
#SBATCH -J compute_mismatch_I_td # Job name
#SBATCH --account=PHY24021 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks=1 # Single task (not MPI)
#SBATCH --cpus-per-task=128 # Use all CPUs on node
#SBATCH --mem=0 # Use all available memory on node
#SBATCH -t 00:10:00 # Walltime
#SBATCH -p development # Queue name where job is submitted
#SBATCH -o batch_outputs/compute_mismatch_I_td_%j.out # Output file
#SBATCH --mail-type=FAIL # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=tvn6@cornell.edu

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/batch_scripts"
source "$SCRIPT_DIR/_common.sh"
source "$SCRIPT_DIR/_contour_I_td_config.sh"

# Requested trial configuration: system 3 (random), mcz=20, I=0.2..0.3 step 0.1, td=20..30 ms step 5.
export MCZ=20
export ORIENT_PRESET=Taman_random
export I_MIN=0.2
export I_MAX=0.3
export I_STEP=0.1
export TD_MIN_MS=20
export TD_MAX_MS=30
export TD_STEP_MS=5
export BANK_DIR=/work/10000/fairytien33/gw_shared_data/template_banks

python -m scripts.mismatch_I_td.compute_mismatch_cubes \
  --mcz "${MCZ}" \
  --orient_preset "${ORIENT_PRESET}" \
  --z "${Z}" \
  $(build_I_grid_args) \
  $(build_td_grid_args) \
  --omega_min "${OMEGA_MIN}" --omega_max "${OMEGA_MAX}" --omega_pts "${OMEGA_PTS}" \
  --theta_min "${THETA_MIN}" --theta_max "${THETA_MAX}" --theta_pts "${THETA_PTS}" \
  --gamma_pts "${GAMMA_PTS}" \
  --n_workers ${NWORKERS} \
  --use_opt_match \
  --bank_dir "${BANK_DIR}" \
  --run_dir "${RUN_DIR}" \
  --I_chunk_index 0 \
  --I_chunk_count 1
