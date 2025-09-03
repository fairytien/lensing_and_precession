#!/bin/bash
#SBATCH -J indiv_otf_v4prec_LSODA_51x101_lowtol # Job name
#SBATCH --account=PHY24021 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes and cores per node required
#SBATCH -t 00:05:00 # Walltime
#SBATCH -p development # Queue name where job is submitted
#SBATCH -o batch_outputs/v4_indiv_otf_LSODA_51x101_lowtol.out # Output file
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=tien.nguyen@utdallas.edu

cd $SLURM_SUBMIT_DIR
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fairytien_gw

srun python scripts/v4_indiv_contour_otf.py --ivp_method LSODA --omega_min 3 --omega_max 4 --omega_points 51 --theta_min 6 --theta_max 10 --theta_points 101 --compare_both --tag res_51x101_lowtol