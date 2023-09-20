#!/bin/bash

for N in 32; do
for K in 1 2 4 8; do
for strat in NN NN_withTabu hillClimber IteratedhillClimber tabu; do

sbatch runscript_array_job.sh  $strat $N $K
#python NKL_HC.py $strat 32 $K

done
done
done