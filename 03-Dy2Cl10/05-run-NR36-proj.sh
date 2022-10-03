#!/bin/bash

#SBATCH --job-name 04-05NP36
#SBATCH -o 05-proj-NR36-%A-%a.out
#SBATCH --array=0,2-5
#SBATCH --nodes=2
#SBATCH --time=7-0:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=90G
#SBATCH --no-requeue

NRT=36
XX=$(expr ${SLURM_ARRAY_TASK_ID} '*' 2)

export PYSCF_TMPDIR=/central/scratch/hczhai/soc-pyscf-tmp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
orterun --bind-to core --map-by ppr:$SLURM_TASKS_PER_NODE:node:pe=$OMP_NUM_THREADS python3 -u 05-dy-20o-dzbp-proj-th28.py ${XX} ${NRT}
