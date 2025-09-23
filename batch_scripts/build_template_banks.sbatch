#!/bin/bash
#SBATCH -J build_banks # Job name
#SBATCH --account=PHY24021 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes
#SBATCH -t 04:00:00 # Walltime
#SBATCH -p normal # Queue name where job is submitted
#SBATCH -o batch_outputs/build_template_banks.out # Output file
#SBATCH --mail-type=BEGIN,FAIL # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=tvn6@cornell.edu

cd $SLURM_SUBMIT_DIR

# Activate conda environment
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gw

# Configurable parameters via environment variables (with sensible defaults)
ORIENT_PRESET=${ORIENT_PRESET:-Taman_edgeon}

MCZ_MIN=${MCZ_MIN:-10}
MCZ_MAX=${MCZ_MAX:-80}
MCZ_PTS=${MCZ_PTS:-71}

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

# Optional override for worker count (defaults to auto inside the script)
if [[ -z "$N_WORKERS" ]]; then
  srun python scripts/build_template_banks.py \
    --orient_preset "$ORIENT_PRESET" \
    --mcz_min $MCZ_MIN --mcz_max $MCZ_MAX --mcz_pts $MCZ_PTS \
    --omega_min $OMEGA_MIN --omega_max $OMEGA_MAX --omega_pts $OMEGA_PTS \
    --theta_min $THETA_MIN --theta_max $THETA_MAX --theta_pts $THETA_PTS \
    --gamma_pts $GAMMA_PTS \
    --f_min $F_MIN --delta_f $DELTA_F \
    --bank_dir "$BANK_DIR" --bank_prefix "$BANK_PREFIX"
else
  srun python scripts/build_template_banks.py \
    --orient_preset "$ORIENT_PRESET" \
    --mcz_min $MCZ_MIN --mcz_max $MCZ_MAX --mcz_pts $MCZ_PTS \
    --omega_min $OMEGA_MIN --omega_max $OMEGA_MAX --omega_pts $OMEGA_PTS \
    --theta_min $THETA_MIN --theta_max $THETA_MAX --theta_pts $THETA_PTS \
    --gamma_pts $GAMMA_PTS \
    --f_min $F_MIN --delta_f $DELTA_F \
    --bank_dir "$BANK_DIR" --bank_prefix "$BANK_PREFIX" \
    --n_workers $N_WORKERS
fi


