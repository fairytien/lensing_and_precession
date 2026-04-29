#!/bin/bash
#SBATCH -J compute_mismatch_I_td_cubes # Job name
#SBATCH --account=PHY24021 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks=1 # Single task (not MPI)
#SBATCH --cpus-per-task=128 # Use all CPUs on node
#SBATCH --mem=0 # Use all available memory on node
#SBATCH -t 00:05:00 # Walltime
#SBATCH -p development # Queue name where job is submitted
#SBATCH --array=0 # Single job (no array)
#SBATCH -o batch_outputs/compute_mismatch_I_td_cubes_test_%A_%a.out # Output file
#SBATCH --mail-type=FAIL # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=tvn6@cornell.edu

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/batch_scripts"
source "$SCRIPT_DIR/_common.sh"
source "$SCRIPT_DIR/_contour_I_td_config.sh"

ORIENT_PRESET=Taman_edgeon
MCZ=25
I_MIN=0.5
I_MAX=0.5
I_PTS=1

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
  --I_chunk_index ${SLURM_ARRAY_TASK_ID} \
  --I_chunk_count ${SLURM_ARRAY_TASK_COUNT}