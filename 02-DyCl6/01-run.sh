#!/bin/bash

#SBATCH --job-name 05-01-DY
#SBATCH -o 01-run-%J.out
#SBATCH --nodes=1
#SBATCH --time=0-1:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=100G
#SBATCH --partition=expansion
#SBATCH --no-requeue

export PYSCF_TMPDIR=/central/scratch/hczhai/soc-pyscf-tmp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python3 -u 01-dy-7o-dzbp-mf-2step-sr42.py
