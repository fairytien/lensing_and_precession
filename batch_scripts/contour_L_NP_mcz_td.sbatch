#!/bin/bash
#SBATCH -J contour_L_NP_mcz_td # Job name
#SBATCH --account=PHY24021 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks=1 # Single task (not MPI)
#SBATCH --cpus-per-task=81 # Use all CPUs on node
#SBATCH --mem=0 # Use all available memory on node
#SBATCH -t 00:00:30 # Walltime
#SBATCH -p development # Queue name where job is submitted
#SBATCH -o batch_outputs/contour_L_NP_mcz_td_%A.out # Output file
#SBATCH --mail-type=FAIL # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=tvn6@cornell.edu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
source "$SCRIPT_DIR/_contour_mcz_td_config.sh"

# Set thread limits to prevent oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

# Keep grid sizing configurable and consistent with worker cap
MCZ_POINTS=${MCZ_POINTS:-${MCZ_PTS}}
TD_POINTS=${TD_POINTS:-${TD_PTS}}

# Use allocated CPUs, but do not exceed number of mcz rows to compute
ALLOC_CPUS=${SLURM_CPUS_PER_TASK:-64}
DEFAULT_WORKERS=$(( ALLOC_CPUS < MCZ_POINTS ? ALLOC_CPUS : MCZ_POINTS ))
NWORKERS=${NWORKERS:-$DEFAULT_WORKERS}

# Run the contour script with all available CPUs
python -m scripts.super_contours.contour_L_NP_mcz_td \
  --I "${FLUX_RATIO}" \
  --mcz_min "${MCZ_MIN}" \
  --mcz_max "${MCZ_MAX}" \
  --mcz_points ${MCZ_POINTS} \
  --td_min_ms "${TD_MIN_MS}" \
  --td_max_ms "${TD_MAX_MS}" \
  --td_points "${TD_POINTS}" \
  --f_min 20.0 \
  --delta_f 0.25 \
  --n_processes ${NWORKERS} \
  --redshift "${Z}"
