#!/bin/bash

#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem 16G
#SBATCH --time 10:00:00 
#SBATCH --output extract_frames.log
#SBATCH -A visualai

source activate cos529 
cd /n/fs/visualai-scr/Data/BWC
python extract_frames.py
