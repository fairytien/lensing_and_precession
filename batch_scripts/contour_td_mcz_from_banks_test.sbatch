#!/bin/bash
#SBATCH -J contour_td_mcz_test # Job name
#SBATCH --account=PHY24021 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks=1 # Single task (not MPI)
#SBATCH --cpus-per-task=128 # Use all CPUs on node
#SBATCH --mem=0 # Use all available memory on node
#SBATCH -t 00:10:00 # Walltime (shorter for test runs)
#SBATCH -p development # Queue name where job is submitted (faster queue)
#SBATCH --array=0 # Single job (no array)
#SBATCH -o batch_outputs/contour_td_mcz_from_banks_test_%A_%a.out # Output file
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
NWORKERS=${NWORKERS:-${SLURM_CPUS_PER_TASK:-64}}

python scripts/contour_td_mcz_from_banks.py \
  --I 0.5 \
  --orient_preset Taman_edgeon \
  --mcz_min 10 --mcz_max 12 --mcz_pts 2 \
  --td_min_ms 20 --td_max_ms 22 --td_pts 3 \
  --omega_min 0 --omega_max 6 --omega_pts 21 \
  --theta_min 0 --theta_max 15 --theta_pts 51 \
  --gamma_pts 51 \
  --f_min 20 --delta_f 0.25 \
  --results_dir "$SLURM_SUBMIT_DIR/data/contours" \
  --n_workers ${NWORKERS} \
  # --mcz_chunk_index ${SLURM_ARRAY_TASK_ID} \
  # --mcz_chunk_count ${SLURM_ARRAY_TASK_COUNT}
