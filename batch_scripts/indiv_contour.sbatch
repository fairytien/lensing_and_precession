#!/bin/bash
#SBATCH -J indiv # Job name
#SBATCH --account=AST23010 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes and cores per node required
#SBATCH -t 00:02:00 # Walltime
#SBATCH -p development # Queue name where job is submitted
#SBATCH -o batch_outputs/v3_indiv_.out # Output file
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=tien.nguyen@utdallas.edu

cd $SLURM_SUBMIT_DIR
conda init
conda activate fairytien_gw

# Specify the mcz value
MCZ_VALUE=40
export MCZ_VALUE

# Load the RP template grid
WORK_DIR=${WORK:-.}
TEMPLATE_GRID_PATH="$WORK_DIR/data/sys2_template_grid_mcz${MCZ_VALUE}.npz"
export TEMPLATE_GRID_PATH

srun python scripts/v3_indiv_contour.py