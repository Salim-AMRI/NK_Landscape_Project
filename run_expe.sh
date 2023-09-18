#!/bin/bash


for K in 8 1 2 4 6  10 12; do

sbatch runscript_array_job.sh 64 $K

done


