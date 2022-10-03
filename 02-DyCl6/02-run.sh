#!/bin/bash

#SBATCH --job-name 05-02-DY
#SBATCH -o 02-run-%J.out
#SBATCH --nodes=1
#SBATCH --time=0-1:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=100G
#SBATCH --partition=expansion
#SBATCH --no-requeue

export PYSCF_TMPDIR=/central/scratch/hczhai/soc-pyscf-tmp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
orterun --bind-to core --map-by ppr:$SLURM_TASKS_PER_NODE:node:pe=$OMP_NUM_THREADS python3 -u 02-dy-7o-dzbp-1step-sr42.py
