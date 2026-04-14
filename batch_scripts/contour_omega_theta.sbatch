#!/bin/bash
#SBATCH -J indiv_otf_v3 # Job name
#SBATCH --account=PHY24021 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes and cores per node required
#SBATCH -t 00:01:00 # Walltime
#SBATCH -p development # Queue name where job is submitted
#SBATCH -o batch_outputs/v3_indiv_otf.out # Output file
#SBATCH --mail-type=NONE # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=tien.nguyen@utdallas.edu

cd $SLURM_SUBMIT_DIR
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fairytien_gw

# Compute angles once (more efficient than multiple Python calls)
read theta_S phi_S theta_J phi_J <<< $(python -c "import numpy as np; print(np.pi / 4, 0, np.pi / 2, np.pi / 2)")

# Run the contour computation
srun python -m scripts.contour_omega_theta.v3_indiv_contour_otf \
    --mcz_msun 29 \
    --td_ms 50.0 \
    --I 0.5 \
    --theta_S "$theta_S" \
    --phi_S "$phi_S" \
    --theta_J "$theta_J" \
    --phi_J "$phi_J" \
    --omega_min 0.0 \
    --omega_max 5.0 \
    --omega_points 51 \
    --theta_min 0.0 \
    --theta_max 15.0 \
    --theta_points 151 \
    --f_min 20.0 \
    --delta_f 0.25 \
    --redshift 1 \
    --use_opt_match