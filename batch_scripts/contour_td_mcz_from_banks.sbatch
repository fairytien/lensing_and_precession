#!/bin/bash
#SBATCH -J contour_td_mcz # Job name
#SBATCH --account=PHY24021 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes
#SBATCH -t 02:00:00 # Walltime
#SBATCH -p normal # Queue name where job is submitted
#SBATCH -o batch_outputs/contour_td_mcz_from_banks.out # Output file
#SBATCH --mail-type=BEGIN,FAIL # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=tvn6@cornell.edu

cd $SLURM_SUBMIT_DIR

# Activate conda environment
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fairytien_gw

# Configurable parameters via environment variables (with sensible defaults)
ORIENT_PRESET=${ORIENT_PRESET:-Taman_edgeon}

I=${I:-0.5}

MCZ_MIN=${MCZ_MIN:-10}
MCZ_MAX=${MCZ_MAX:-80}
MCZ_PTS=${MCZ_PTS:-71}

TD_MIN_MS=${TD_MIN_MS:-20}
TD_MAX_MS=${TD_MAX_MS:-70}
TD_PTS=${TD_PTS:-51}

OMEGA_MIN=${OMEGA_MIN:-0}
OMEGA_MAX=${OMEGA_MAX:-6}
OMEGA_PTS=${OMEGA_PTS:-61}

THETA_MIN=${THETA_MIN:-0}
THETA_MAX=${THETA_MAX:-15}
THETA_PTS=${THETA_PTS:-151}

GAMMA_PTS=${GAMMA_PTS:-101}

F_MIN=${F_MIN:-20}
DELTA_F=${DELTA_F:-0.25}

BANK_DIR=${BANK_DIR:-$SLURM_SUBMIT_DIR/data/template_banks}
BANK_PREFIX=${BANK_PREFIX:-rp_bank}
RESULTS_DIR=${RESULTS_DIR:-$SLURM_SUBMIT_DIR/data/contours}

# Optional flags
COMPARE_BOTH=${COMPARE_BOTH:-false}
USE_OPT_MATCH=${USE_OPT_MATCH:-false}
SAVE_FULL_MISMATCH=${SAVE_FULL_MISMATCH:-false}
NO_PLOT=${NO_PLOT:-false}

# Build command with conditional flags
CMD="srun python scripts/contour_td_mcz_from_banks.py \
  --I $I \
  --orient_preset \"$ORIENT_PRESET\" \
  --mcz_min $MCZ_MIN --mcz_max $MCZ_MAX --mcz_pts $MCZ_PTS \
  --td_min_ms $TD_MIN_MS --td_max_ms $TD_MAX_MS --td_pts $TD_PTS \
  --omega_min $OMEGA_MIN --omega_max $OMEGA_MAX --omega_pts $OMEGA_PTS \
  --theta_min $THETA_MIN --theta_max $THETA_MAX --theta_pts $THETA_PTS \
  --gamma_pts $GAMMA_PTS \
  --f_min $F_MIN --delta_f $DELTA_F \
  --bank_dir \"$BANK_DIR\" --bank_prefix \"$BANK_PREFIX\" \
  --results_dir \"$RESULTS_DIR\""

# Add optional flags if enabled
if [[ "$COMPARE_BOTH" == "true" ]]; then
  CMD="$CMD --compare_both"
fi

if [[ "$USE_OPT_MATCH" == "true" ]]; then
  CMD="$CMD --use_opt_match"
fi

if [[ "$SAVE_FULL_MISMATCH" == "true" ]]; then
  CMD="$CMD --save_full_mismatch"
fi

if [[ "$NO_PLOT" == "true" ]]; then
  CMD="$CMD --no_plot"
fi

# Optional override for worker count (defaults to auto inside the script)
if [[ -n "$N_WORKERS" ]]; then
  CMD="$CMD --n_workers $N_WORKERS"
fi

# Execute the command
eval $CMD
