#!/bin/bash
#SBATCH -J indiv_otf_v2prec_101x401 # Job name
#SBATCH --account=PHY24021 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes and cores per node required
#SBATCH -t 00:20:00 # Walltime
#SBATCH -p normal # Queue name where job is submitted
#SBATCH -o batch_outputs/v3_indiv_otf_v2prec_res_omega101_theta401.out # Output file
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
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

srun python scripts/v3_indiv_contour_otf_v2prec.py --omega_min 3 --omega_max 4 --omega_points 101 --theta_min 6 --theta_max 10 --theta_points 401 --compare_both --tag res_omega101_theta401