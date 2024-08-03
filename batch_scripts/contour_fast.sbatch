#!/bin/bash
#SBATCH -J superNP_40 # Job name
#SBATCH --account=AST23010 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes and cores per node required
#SBATCH -t 00:10:00 # Walltime
#SBATCH -p normal # Queue name where job is submitted
#SBATCH -o batch_outputs/superNP_40.out # Output file
#SBATCH --mail-type=BEGIN # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=tien.nguyen@utdallas.edu

# Print the task and run range
echo This is task $SLURM_ARRAY_TASK_ID

cd $SLURM_SUBMIT_DIR

# module load anaconda3
conda init
conda activate fairytien_gw

srun python scripts/super_contour_L_NP.py