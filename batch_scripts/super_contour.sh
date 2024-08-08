#!/bin/bash

# Specify the mcz value
MCZ_VALUE=40
export MCZ_VALUE

# Load the RP template grid
WORK_DIR=${WORK:-.}
TEMPLATE_GRID_PATH="$WORK_DIR/data/sys2_template_grid_mcz${MCZ_VALUE}.npz"
export TEMPLATE_GRID_PATH

# Submit the job with dynamic job name and output file
sbatch --job-name=super${MCZ_VALUE} --output=batch_outputs/v3_super${MCZ_VALUE}.out <<EOF
#!/bin/bash
#SBATCH --account=AST23010 # Account to submit jobs under
#SBATCH -N 1 # Number of nodes and cores per node required
#SBATCH -t 00:15:00 # Walltime
#SBATCH -p development # Queue name where job is submitted
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=tien.nguyen@utdallas.edu

echo This is task \$SLURM_ARRAY_TASK_ID
cd \$SLURM_SUBMIT_DIR
conda init
conda activate fairytien_gw

srun python scripts/v3_super_contour.py
EOF