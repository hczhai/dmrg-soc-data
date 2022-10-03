#!/bin/bash

#SBATCH --job-name 04-01MF
#SBATCH -o 01-mf-%J.out
#SBATCH --nodes=1
#SBATCH --time=1-0:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=370G
#SBATCH --no-requeue

export PYSCF_TMPDIR=/central/scratch/hczhai/soc-pyscf-tmp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python3 -u 01-dy-20o-dzbp-mf-2step.py
