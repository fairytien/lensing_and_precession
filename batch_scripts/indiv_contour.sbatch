#!/bin/bash
#SBATCH -J indiv_otf_v2prec # Job name
#SBATCH --account=PHY24021 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes and cores per node required
#SBATCH -t 00:15:00 # Walltime
#SBATCH -p development # Queue name where job is submitted
#SBATCH -o batch_outputs/v3_indiv_otf_v2prec.out # Output file
#SBATCH --mail-type=NONE # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=tien.nguyen@utdallas.edu

cd $SLURM_SUBMIT_DIR
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fairytien_gw

# # Specify the mcz value
# MCZ_VALUE=40
# export MCZ_VALUE

# # Load the RP template grid
# WORK_DIR=${WORK:-.}
# TEMPLATE_GRID_PATH="$WORK_DIR/data/sys2_template_grid_mcz${MCZ_VALUE}.npz"
# export TEMPLATE_GRID_PATH

# Compute angles once (more efficient than multiple Python calls)
read theta_S phi_S theta_J phi_J <<< $(python -c "import numpy as np; print(np.pi / 4, 0, 8 * np.pi / 9, np.pi / 4)")

# Run the contour computation
srun python scripts/individual_contours/v3_indiv_contour_otf_v2prec.py \
    --mcz_msun 40 \
    --td_ms 30 \
    --I 0.5 \
    --theta_S "$theta_S" \
    --phi_S "$phi_S" \
    --theta_J "$theta_J" \
    --phi_J "$phi_J" \
    --omega_min 0 \
    --omega_max 4 \
    --omega_points 81 \
    --theta_min 0 \
    --theta_max 15 \
    --theta_points 151 \
    --compare_both \
    --tag res81x151