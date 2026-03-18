#!/bin/bash
#SBATCH -J build_banks_test # Job name
#SBATCH --account=PHY24021 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks=1 # Single task (not MPI)
#SBATCH --cpus-per-task=128 # Use all CPUs on node
#SBATCH --mem=0 # Use all available memory on node
#SBATCH -t 00:12:00 # Walltime (estimated for higher resolution runs)
#SBATCH -p development # Queue name where job is submitted (faster queue)
#SBATCH --array=0 # Single job (no array)
#SBATCH -o batch_outputs/build_template_banks_test_%A_%a.out # Output file
#SBATCH --mail-type=FAIL # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=tvn6@cornell.edu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
source "$SCRIPT_DIR/_contour_td_mcz_config.sh"

# Use all allocated CPUs for multiprocessing workers
NWORKERS=${NWORKERS:-32}

ORIENT_PRESET=Taman_random
MCZ_MIN=30
MCZ_MAX=30
MCZ_PTS=1
GAMMA_PTS=101

python -m scripts.template_banks.build_template_banks \
  --orient_preset "${ORIENT_PRESET}" \
  --mcz_min "${MCZ_MIN}" --mcz_max "${MCZ_MAX}" --mcz_pts "${MCZ_PTS}" \
  --omega_min "${OMEGA_MIN}" --omega_max "${OMEGA_MAX}" --omega_pts "${OMEGA_PTS}" \
  --theta_min "${THETA_MIN}" --theta_max "${THETA_MAX}" --theta_pts "${THETA_PTS}" \
  --gamma_pts "${GAMMA_PTS}" \
  --n_workers ${NWORKERS} \
  --dtype complex64 \
  --bank_dir "${BANK_DIR}" \
  --mcz_chunk_index ${SLURM_ARRAY_TASK_ID} \
  --mcz_chunk_count ${SLURM_ARRAY_TASK_COUNT}
