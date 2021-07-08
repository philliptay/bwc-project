#!/bin/bash

for chunk_num in 0 1 2 3 4 5 6 7 8 9
do
    sbatch --exclude=node718 detect_faces.slurm ${chunk_num} 
done
