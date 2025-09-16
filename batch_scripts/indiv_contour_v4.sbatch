#!/bin/bash
#SBATCH -J indiv_otf_v4prec_DOP853_101x101 # Job name
#SBATCH --account=PHY24021 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes and cores per node required
#SBATCH -t 00:05:00 # Walltime
#SBATCH -p development # Queue name where job is submitted
#SBATCH -o batch_outputs/v4_indiv_otf_DOP853_101x101.out # Output file
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=tien.nguyen@utdallas.edu

cd $SLURM_SUBMIT_DIR
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fairytien_gw

srun python scripts/v4_indiv_contour_otf.py --ivp_method DOP853 --omega_min 3 --omega_max 4 --omega_points 101 --theta_min 6 --theta_max 10 --theta_points 101 --compare_both --tag res_101x101