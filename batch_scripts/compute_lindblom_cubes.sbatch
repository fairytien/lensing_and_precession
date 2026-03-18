#!/bin/bash
#SBATCH -J compute_lindblom # Job name
#SBATCH --account=PHY24021 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks=1 # Single task (not MPI)
#SBATCH --cpus-per-task=128 # Use all CPUs on node
#SBATCH --mem=0 # Use all available memory on node
#SBATCH -t 00:20:00 # Walltime (20 minutes per cube)
#SBATCH -p normal # Queue name where job is submitted
#SBATCH --array=0-80 # Array job for mcz values 10-90 (81 total)
#SBATCH -o batch_outputs/compute_lindblom_%A_%a.out # Output file
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

# Map array index to mcz value (10-90)
MCZ_VAL=$((10 + SLURM_ARRAY_TASK_ID))

# Paths
CUBE_PATH="data/contours_td_mcz/mismatch_cubes/mismatch_cubes_mcz${MCZ_VAL}Msun_td20-70ms_Taman_edgeon.h5"
BANK_DIR="data/template_banks"
BANK_PREFIX="rp_bank"
OUTPUT_DIR="data/contours_td_mcz/mismatch_cubes"

# Check if cube exists
if [ ! -f "$CUBE_PATH" ]; then
  echo "Error: Mismatch cube not found: $CUBE_PATH"
  exit 1
fi

# Check if Lindblom cube already exists with SNR data
LINDBLOM_PATH="${CUBE_PATH%.h5}_lindblom.h5"
if [ -f "$LINDBLOM_PATH" ]; then
  # Check if it has SNR data
  python -c "import h5py; h5 = h5py.File('$LINDBLOM_PATH', 'r'); has_snr = 'snr_cube' in h5; h5.close(); exit(0 if has_snr else 1)" 2>/dev/null
  if [ $? -eq 0 ]; then
    echo "Lindblom cube with SNR already exists for mcz=$MCZ_VAL, skipping..."
    exit 0
  fi
fi

echo "Processing mcz=$MCZ_VAL (array task ${SLURM_ARRAY_TASK_ID})..."
echo "Cube path: $CUBE_PATH"
echo "Using $NWORKERS workers"

# Run the computation
python -m scripts.lindblom.compute_lindblom_contours \
  --cube_path "$CUBE_PATH" \
  --bank_dir "$BANK_DIR" \
  --bank_prefix "$BANK_PREFIX" \
  --f_min 20.0 \
  --delta_f 0.25 \
  --n_workers "$NWORKERS"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "Successfully computed Lindblom cube for mcz=$MCZ_VAL"
else
  echo "Error computing Lindblom cube for mcz=$MCZ_VAL (exit code: $EXIT_CODE)"
  exit $EXIT_CODE
fi

