#!/bin/bash
#SBATCH -J compute_mismatch # Job name
#SBATCH --account=PHY24021 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks=1 # Single task (not MPI)
#SBATCH --cpus-per-task=128 # Use all CPUs on node
#SBATCH --mem=0 # Use all available memory on node
#SBATCH -t 00:12:00 # Walltime
#SBATCH -p normal # Queue name where job is submitted
#SBATCH --array=0-9 # Number of chunks
#SBATCH -o batch_outputs/compute_mismatch_%A_%a.out # Output file
#SBATCH --mail-type=FAIL # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=tvn6@cornell.edu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
source "$SCRIPT_DIR/_contour_td_mcz_config.sh"

# Use all allocated CPUs for multiprocessing workers
NWORKERS=${NWORKERS:-${SLURM_CPUS_PER_TASK:-64}}

# NOTE: This array job computes per-mcz mismatch cubes in chunks.
# After ALL array tasks complete, run:
# 1. python scripts/mismatch_mcz_td/aggregate_best_match.py --results_dir "$RESULTS_DIR" --td_min_ms "$TD_MIN_MS" --td_max_ms "$TD_MAX_MS" --mcz_min "$MCZ_MIN" --mcz_max "$MCZ_MAX" --orientation_tag "$ORIENT_PRESET"
# 2. python scripts/mismatch_mcz_td/create_contour_mcz_td_from_best_match.py --results_dir "$RESULTS_DIR" --td_min_ms "$TD_MIN_MS" --td_max_ms "$TD_MAX_MS" --mcz_min "$MCZ_MIN" --mcz_max "$MCZ_MAX" --orientation_tag "$ORIENT_PRESET"

python scripts/mismatch_mcz_td/compute_mismatch_cubes.py \
  --I "${FLUX_RATIO}" \
  --orient_preset "${ORIENT_PRESET}" \
  --mcz_min "${MCZ_MIN}" --mcz_max "${MCZ_MAX}" --mcz_pts "${MCZ_PTS}" \
  --td_min_ms "${TD_MIN_MS}" --td_max_ms "${TD_MAX_MS}" --td_pts "${TD_PTS}" \
  --omega_min "${OMEGA_MIN}" --omega_max "${OMEGA_MAX}" --omega_pts "${OMEGA_PTS}" \
  --theta_min "${THETA_MIN}" --theta_max "${THETA_MAX}" --theta_pts "${THETA_PTS}" \
  --gamma_pts "${GAMMA_PTS}" \
  --n_workers ${NWORKERS} \
  --use_opt_match \
  --bank_dir "${BANK_DIR}" \
  --results_dir "${RESULTS_DIR}" \
  --mcz_chunk_index ${SLURM_ARRAY_TASK_ID} \
  --mcz_chunk_count ${SLURM_ARRAY_TASK_COUNT}

