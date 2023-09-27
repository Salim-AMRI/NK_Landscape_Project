#!/bin/bash

#SBATCH --array=0-9
#SBATCH --time=8:00:00
#SBATCH -N1
#SBATCH --no-kill
#SBATCH --error=slurm-err-%j.out
#SBATCH --output=slurm-o-%j.out	
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10000
#SBATCH --cpus-per-task=20	
#SBATCH  -p SMP-short


python NKL_HC.py $1 $2 $3 --seed $SLURM_ARRAY_TASK_ID
