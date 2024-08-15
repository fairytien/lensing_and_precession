#!/bin/bash
#SBATCH -J contours_mcz # Job name
#SBATCH --account=AST23010 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes and cores per node required
#SBATCH -t 01:00:00 # Walltime
#SBATCH -p development # Queue name where job is submitted
#SBATCH -o batch_outputs/contours_mcz.out # Output file
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=tien.nguyen@utdallas.edu

cd $SLURM_SUBMIT_DIR

conda init
conda activate fairytien_gw

srun python scripts/v3_contours_mcz.py --output_dir $WORK/data