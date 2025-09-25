#!/bin/bash
#SBATCH -J contour_td_mcz # Job name
#SBATCH --account=PHY24021 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks=1 # Single task (not MPI)
#SBATCH --cpus-per-task=128 # Use all CPUs on node
#SBATCH --mem=0 # Use all available memory on node
#SBATCH -t 04:00:00 # Walltime
#SBATCH -p normal # Queue name where job is submitted
#SBATCH --array=0-9 # Number of chunks
#SBATCH -o batch_outputs/contour_td_mcz_from_banks_%A_%a.out # Output file
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

# NOTE: This array job computes mismatches in chunks. Each chunk writes a partial best_match file.
# After ALL array tasks complete, run: python scripts/aggregate_best_match.py --results_dir ./data/contours --td_min_ms 20 --td_max_ms 70 --mcz_min 10 --mcz_max 80 --orientation_tag Taman_edgeon

python scripts/contour_td_mcz_from_banks.py \
  --I 0.5 \
  --orient_preset Taman_edgeon \
  --mcz_min 10 --mcz_max 80 --mcz_pts 71 \
  --td_min_ms 20 --td_max_ms 70 --td_pts 51 \
  --omega_min 0 --omega_max 6 --omega_pts 61 \
  --theta_min 0 --theta_max 15 --theta_pts 151 \
  --gamma_pts 101 \
  --n_workers ${NWORKERS} \
  --use_opt_match \
  --results_dir ./data/contours \
  --no_plot \
  --mcz_chunk_index ${SLURM_ARRAY_TASK_ID} \
  --mcz_chunk_count ${SLURM_ARRAY_TASK_COUNT}
