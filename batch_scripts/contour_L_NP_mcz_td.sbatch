#!/bin/bash
#SBATCH -J contour_L_NP_mcz_td # Job name
#SBATCH --account=PHY24021 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks=1 # Single task (not MPI)
#SBATCH --cpus-per-task=128 # Use all CPUs on node
#SBATCH --mem=0 # Use all available memory on node
#SBATCH -t 00:10:00 # Walltime
#SBATCH -p development # Queue name where job is submitted
#SBATCH -o batch_outputs/contour_L_NP_mcz_td_%A.out # Output file
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
export MPLBACKEND=Agg

# Keep grid sizing configurable and consistent with worker cap
MCZ_POINTS=${MCZ_POINTS:-81}

# Use allocated CPUs, but do not exceed number of mcz rows to compute
ALLOC_CPUS=${SLURM_CPUS_PER_TASK:-64}
DEFAULT_WORKERS=$(( ALLOC_CPUS < MCZ_POINTS ? ALLOC_CPUS : MCZ_POINTS ))
NWORKERS=${NWORKERS:-$DEFAULT_WORKERS}

# Run the contour script with all available CPUs
python scripts/super_contours/contour_L_NP_mcz_td.py \
  --I 0.5 \
  --mcz_min 10.0 \
  --mcz_max 90.0 \
  --mcz_points ${MCZ_POINTS} \
  --td_min_ms 20.0 \
  --td_max_ms 70.0 \
  --td_points 51 \
  --f_min 20.0 \
  --delta_f 0.25 \
  --n_processes ${NWORKERS} \
  --redshift 0.1 \
  --compare_both

