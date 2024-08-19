#!/bin/bash

for N in 32 64 128; do
for K in 1 2 4 8; do
for seed in 0 1 2 3 4 5 6 7 8 9; do

python Main.py hillClimberJump NK $N $K --seed $seed

done
done
done


