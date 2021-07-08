#!/bin/bash

#SBATCH --job-name=detect_faces # the name of the job
#SBATCH --output=outputs/%j.log # where stdout and stderr will write to
#SBATCH -A visualai
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1                   # number of GPUs your job requests
#SBATCH --mem=32G                      # amount of memory needed
#SBATCH --time=48:00:00       

CHUNKNUM=$1

hostname
source activate cos529
cd /n/fs/vai-bwc-scr/methods/face_detection/
python detect_faces.py --chunk_num ${CHUNKNUM}
