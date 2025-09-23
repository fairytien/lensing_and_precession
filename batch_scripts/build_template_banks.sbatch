#!/bin/bash
#SBATCH -J build_banks # Job name
#SBATCH --account=PHY24021 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes
#SBATCH -t 04:00:00 # Walltime
#SBATCH -p normal # Queue name where job is submitted
#SBATCH --ntasks-per-node=128 # Number of tasks per node
#SBATCH --array=0-9 # Number of chunks
#SBATCH -o batch_outputs/build_template_banks.out # Output file
#SBATCH --mail-type=BEGIN,FAIL # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=tvn6@cornell.edu

# Activate your conda environment for this project
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
if command -v conda >/dev/null 2>&1; then
  conda activate fairytien_gw 2>/dev/null || { source ~/.bashrc && conda activate fairytien_gw; }
fi

# Change to submission directory or PROJECT_ROOT if provided
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"

# Tune n_workers to available cores per node; default to a conservative value
NWORKERS=${NWORKERS:-${SLURM_CPUS_ON_NODE:-96}}

python scripts/build_template_banks.py \
  --orient_preset edgeon \
  --mcz_min 10 --mcz_max 80 --mcz_pts 71 \
  --omega_min 0 --omega_max 6 --omega_pts 61 \
  --theta_min 0 --theta_max 15 --theta_pts 151 \
  --gamma_pts 101 \
  --f_min 20 --delta_f 0.25 \
  --n_workers ${NWORKERS} \
  --dtype complex64 \
  --mcz_chunk_index ${SLURM_ARRAY_TASK_ID} \
  --mcz_chunk_count ${SLURM_ARRAY_TASK_COUNT}


