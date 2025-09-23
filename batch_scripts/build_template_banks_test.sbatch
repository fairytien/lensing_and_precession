#!/bin/bash
#SBATCH -J build_banks_test # Job name
#SBATCH --account=PHY24021 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes
#SBATCH -t 00:05:00 # Walltime (shorter for test runs)
#SBATCH -p development # Queue name where job is submitted (faster queue)
#SBATCH --array=0-2 # Test with 3 chunks
#SBATCH -o batch_outputs/build_template_banks_test_%A_%a.out # Output file
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

# Test run parameters (smaller grids for quick testing)
ORIENT_PRESET=${ORIENT_PRESET:-Taman_edgeon}

MCZ_MIN=${MCZ_MIN:-10}
MCZ_MAX=${MCZ_MAX:-12}
MCZ_PTS=${MCZ_PTS:-2}

OMEGA_MIN=${OMEGA_MIN:-0}
OMEGA_MAX=${OMEGA_MAX:-6}
OMEGA_PTS=${OMEGA_PTS:-21}

THETA_MIN=${THETA_MIN:-0}
THETA_MAX=${THETA_MAX:-15}
THETA_PTS=${THETA_PTS:-51}

GAMMA_PTS=${GAMMA_PTS:-51}

F_MIN=${F_MIN:-20}
DELTA_F=${DELTA_F:-0.25}

BANK_DIR=${BANK_DIR:-$SLURM_SUBMIT_DIR/data/template_banks}
BANK_PREFIX=${BANK_PREFIX:-rp_bank}

# Data type for complex arrays (complex128 or complex64)
DTYPE=${DTYPE:-complex128}

python scripts/build_template_banks.py \
  --orient_preset "$ORIENT_PRESET" \
  --mcz_min $MCZ_MIN --mcz_max $MCZ_MAX --mcz_pts $MCZ_PTS \
  --omega_min $OMEGA_MIN --omega_max $OMEGA_MAX --omega_pts $OMEGA_PTS \
  --theta_min $THETA_MIN --theta_max $THETA_MAX --theta_pts $THETA_PTS \
  --gamma_pts $GAMMA_PTS \
  --f_min $F_MIN --delta_f $DELTA_F \
  --bank_dir "$BANK_DIR" --bank_prefix "$BANK_PREFIX" \
  --n_workers ${NWORKERS} \
  --dtype "$DTYPE" \
  --mcz_chunk_index ${SLURM_ARRAY_TASK_ID} \
  --mcz_chunk_count ${SLURM_ARRAY_TASK_COUNT}
