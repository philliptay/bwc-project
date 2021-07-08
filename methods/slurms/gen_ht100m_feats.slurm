#!/bin/bash

#SBATCH --job-name=ht100m_feats # the name of the job
#SBATCH --output=outputs/%j.log # where stdout and stderr will write to
#SBATCH -A visualai
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem=32G                      # amount of memory needed
#SBATCH --time=48:00:00       

hostname
source activate cos529
cd /n/fs/vai-bwc-scr/methods/action_detection/video_feature_extractor/
python extract.py --csv=paths.csv --type=3d --batch_size=64
