#!/bin/bash

for N in 32; do
for K in 4 8; do
for strat in  hillClimber tabu IteratedhillClimber NN NN_withTabu; do

sbatch runscript_array_job.sh  $strat $N $K

done
done
done
