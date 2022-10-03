#!/bin/bash

#SBATCH --job-name 01-00-AU
#SBATCH -o 00-run-%J.out
#SBATCH --nodes=1
#SBATCH --time=1-0:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=expansion
#SBATCH --mem=100G
#SBATCH --no-requeue

export PYSCF_TMPDIR=/central/scratch/hczhai/soc-pyscf-tmp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
orterun --bind-to core --map-by ppr:$SLURM_TASKS_PER_NODE:node:pe=$OMP_NUM_THREADS python3 -u 00-au-11o-bp-1step.py
