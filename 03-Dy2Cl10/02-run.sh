#!/bin/bash

#SBATCH --job-name 04-021S
#SBATCH -o 02-1s-%J.out
#SBATCH --nodes=2
#SBATCH --time=7-0:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=370G
#SBATCH --no-requeue

export PYSCF_TMPDIR=/central/scratch/hczhai/soc-pyscf-tmp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
orterun --bind-to core --map-by ppr:$SLURM_TASKS_PER_NODE:node:pe=$OMP_NUM_THREADS python3 -u 02-dy-20o-dzbp-1step.py
