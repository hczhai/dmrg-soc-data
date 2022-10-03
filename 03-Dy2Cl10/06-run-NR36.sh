#!/bin/bash

#SBATCH --job-name 04-06NR36
#SBATCH -o 06-NR36-%J.out
#SBATCH --nodes=1
#SBATCH --qos=debug
#SBATCH --time=0:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --no-requeue

NRT=36

export PYSCF_TMPDIR=/central/scratch/hczhai/soc-pyscf-tmp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python3 -u 06-dy-20o-dzbp-diag-th28.py ${NRT}
