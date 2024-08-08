#!/bin/bash
#SBATCH -J sys2_bank_40 # Job name
#SBATCH --account=AST23010 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes and cores per node required
#SBATCH -t 00:03:00 # Walltime
#SBATCH -p development # Queue name where job is submitted
#SBATCH -o batch_outputs/sys2_bank_40.out # Output file
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=tien.nguyen@utdallas.edu

cd $SLURM_SUBMIT_DIR

conda init
conda activate fairytien_gw

srun python scripts/template_bank_npz.py --output_dir $WORK/data