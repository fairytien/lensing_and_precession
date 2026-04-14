#!/bin/bash
#SBATCH -J build_banks # Job name
#SBATCH --account=PHY24021 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks=1 # Single task (not MPI)
#SBATCH --cpus-per-task=128 # Use all CPUs on node
#SBATCH --mem=0 # Use all available memory on node
#SBATCH -t 00:09:00 # Fallback walltime (prefer tiered submit helper)
#SBATCH -p normal # Queue name where job is submitted
#SBATCH --array=0-80
#SBATCH -o batch_outputs/build_template_banks_%A_%a.out # Output file
#SBATCH --mail-type=FAIL # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=tvn6@cornell.edu

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/batch_scripts"
source "$SCRIPT_DIR/_common.sh"
source "$SCRIPT_DIR/_contour_mcz_td_config.sh"

# For better queue efficiency with mcz-dependent runtimes, submit with:
#   bash batch_scripts/submit_build_template_banks_tiered.sh

python -m scripts.template_banks.build_template_banks \
  --orient_preset "${ORIENT_PRESET}" \
  --z "${Z}" \
  $(build_mcz_grid_args) \
  --omega_min "${OMEGA_MIN}" --omega_max "${OMEGA_MAX}" --omega_pts "${OMEGA_PTS}" \
  --theta_min "${THETA_MIN}" --theta_max "${THETA_MAX}" --theta_pts "${THETA_PTS}" \
  --gamma_pts "${GAMMA_PTS}" \
  --n_workers ${NWORKERS} \
  --dtype complex64 \
  --bank_dir "${BANK_DIR}" \
  --mcz_chunk_index ${SLURM_ARRAY_TASK_ID} \
  --mcz_chunk_count ${SLURM_ARRAY_TASK_COUNT}
