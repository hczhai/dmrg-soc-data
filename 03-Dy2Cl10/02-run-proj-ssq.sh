#!/bin/bash

#SBATCH --job-name 04-021SPS
#SBATCH -o 02-proj-ssq-1s-%J.out
#SBATCH --nodes=1
#SBATCH --time=4-0:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=140G
#SBATCH --no-requeue

export PYSCF_TMPDIR=/central/scratch/hczhai/soc-pyscf-tmp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
orterun --bind-to core --map-by ppr:$SLURM_TASKS_PER_NODE:node:pe=$OMP_NUM_THREADS python3 -u 02-dy-20o-dzbp-1step-proj-ssq.py
