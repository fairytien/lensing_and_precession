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

# Set thread limits to prevent oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Use all allocated CPUs for multiprocessing workers
# NWORKERS=${NWORKERS:-${SLURM_CPUS_PER_TASK:-64}}
NWORKERS=${NWORKERS:-32}


python scripts/template_banks/build_template_banks.py \
  --orient_preset Taman_random \
  --mcz_min 30 --mcz_max 30 --mcz_pts 1 \
  --omega_min 0 --omega_max 6 --omega_pts 61 \
  --theta_min 0 --theta_max 15 --theta_pts 151 \
  --gamma_pts 101 \
  --n_workers ${NWORKERS} \
  --dtype complex64 \
  --mcz_chunk_index ${SLURM_ARRAY_TASK_ID} \
  --mcz_chunk_count ${SLURM_ARRAY_TASK_COUNT}
