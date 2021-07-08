#!/bin/bash                                                                                                                       
#SBATCH --job-name=detect-pose          # the name of the job                                                                      
#SBATCH --output=output.txt            # where stdout and stderr will write to                                                    
#SBATCH -A visualai                                                                                                               
#SBATCH --gres=gpu:1                   # number of GPUs your job requests                                                         
#SBATCH --mem=32G                      # amount of memory needed                                                                  
#SBATCH --time=124:00:00                 # limit on total runtime                                                                 
CHUNKNUM=$1

source ~/.bashrc
conda activate pytorch-openpose
cd /n/fs/vai-bwc-scr/pose_detection
python detect_pose.py --chunk_num ${CHUNKNUM}
