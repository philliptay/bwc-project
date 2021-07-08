#!/bin/bash

for chunk_num in 1 
do
    sbatch --exclude=node718 detect_persons.slurm ${chunk_num} 
done
