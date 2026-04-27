#!/bin/bash
#SBATCH -J compute_mismatch_test # Job name
#SBATCH --account=PHY24021 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1 # One Python rank per node (for node-level mcz chunking)
#SBATCH --cpus-per-task=128 # Use all CPUs on node
#SBATCH --mem=0 # Use all available memory on node
#SBATCH -t 00:09:00 # Walltime
#SBATCH -p development # Queue name where job is submitted
#SBATCH --array=0 # Single job (no array)
#SBATCH -o batch_outputs/compute_mismatch_test_%A_%a.out # Output file
#SBATCH --mail-type=FAIL # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=tvn6@cornell.edu

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/batch_scripts"
source "$SCRIPT_DIR/_common.sh"
source "$SCRIPT_DIR/_contour_mcz_td_config.sh"

ORIENT_PRESET=Taman_faceon
MCZ_MIN=8.5
MCZ_MAX=8.5
MCZ_PTS=1
NODE_COUNT="${SLURM_JOB_NUM_NODES:-1}"
ARRAY_IDX="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_CNT="${SLURM_ARRAY_TASK_COUNT:-1}"

if [ "${NODE_COUNT}" -gt 1 ]; then
  srun --ntasks="${NODE_COUNT}" --ntasks-per-node=1 bash -lc "
    python -m scripts.mismatch_mcz_td.compute_mismatch_cubes \
      --I '${FLUX_RATIO}' \
      --orient_preset '${ORIENT_PRESET}' \
      --z '${Z}' \
      --mcz_min '${MCZ_MIN}' --mcz_max '${MCZ_MAX}' --mcz_pts '${MCZ_PTS}' \
      --td_min_ms '${TD_MIN_MS}' --td_max_ms '${TD_MAX_MS}' --td_pts '${TD_PTS}' \
      --omega_min '${OMEGA_MIN}' --omega_max '${OMEGA_MAX}' --omega_pts '${OMEGA_PTS}' \
      --theta_min '${THETA_MIN}' --theta_max '${THETA_MAX}' --theta_pts '${THETA_PTS}' \
      --gamma_pts '${GAMMA_PTS}' \
      --n_workers '${NWORKERS}' \
      --use_opt_match \
      --bank_dir '${BANK_DIR}' \
      --run_dir '${RUN_DIR}' \
      --mcz_chunk_index '${ARRAY_IDX}' \
      --mcz_chunk_count '${ARRAY_CNT}' \
      --node_chunk_index \"\${SLURM_PROCID}\" \
      --node_chunk_count \"\${SLURM_NTASKS}\"
  "
else
  python -m scripts.mismatch_mcz_td.compute_mismatch_cubes \
    --I "${FLUX_RATIO}" \
    --orient_preset "${ORIENT_PRESET}" \
    --z "${Z}" \
    --mcz_min "${MCZ_MIN}" --mcz_max "${MCZ_MAX}" --mcz_pts "${MCZ_PTS}" \
    --td_min_ms "${TD_MIN_MS}" --td_max_ms "${TD_MAX_MS}" --td_pts "${TD_PTS}" \
    --omega_min "${OMEGA_MIN}" --omega_max "${OMEGA_MAX}" --omega_pts "${OMEGA_PTS}" \
    --theta_min "${THETA_MIN}" --theta_max "${THETA_MAX}" --theta_pts "${THETA_PTS}" \
    --gamma_pts "${GAMMA_PTS}" \
    --n_workers "${NWORKERS}" \
    --use_opt_match \
    --bank_dir "${BANK_DIR}" \
    --run_dir "${RUN_DIR}" \
    --mcz_chunk_index "${ARRAY_IDX}" \
    --mcz_chunk_count "${ARRAY_CNT}" \
    --node_chunk_index 0 \
    --node_chunk_count 1
fi
