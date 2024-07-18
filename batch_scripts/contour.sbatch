#!/bin/bash
#SBATCH -J super_20 # Job name
#SBATCH --account=AST23010 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes and cores per node required
#SBATCH -t 02:20:00 # Walltime
#SBATCH -p normal # Queue name where job is submitted
#SBATCH -o batch_outputs/super_30.out # Output file
#SBATCH --mail-type=BEGIN # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=tien.nguyen@utdallas.edu
#SBATCH --array=0-3

# Print the task and run range
echo This is task $SLURM_ARRAY_TASK_ID

cd $SLURM_SUBMIT_DIR

# module load anaconda3
conda init
conda activate fairytien_gw

srun python scripts/super_contour_mcz20.py