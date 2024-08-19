#!/bin/bash

for N in 64; do
for K in 8; do
for strat in strategyNNRanked_delta_rescale  strategyNNRanked_v1 strategyNNRanked_v2 strategyNNRanked_v1_zScore strategyNNRanked_v2_zScore StrategyNNFitness_and_current strategyNN; do

sbatch runscript_array_job.sh  $strat $N $K

done
done
done
