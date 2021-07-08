#!/bin/bash                                                                                                                               
#SBATCH --job-name=overlay          # the name of the job                                                                             
#SBATCH --exclude=node718
#SBATCH --output=output.txt            # where stdout and stderr will write to                                                            
#SBATCH -A visualai                                                                                                                       
#SBATCH --gres=gpu:1                   # number of GPUs your job requests                                                                 
#SBATCH --mem=32G                      # amount of memory needed                                                                          
#SBATCH --time=124:00:00                 # limit on total runtime                                                                         
CHUNKNUM=$1

source ~/.bashrc
conda activate pytorch-openpose
cd /n/fs/vai-bwc-scr/demo
python TW_overlay_demo.py --chunk_num 0
