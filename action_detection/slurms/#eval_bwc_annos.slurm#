#!/bin/bash                                                                                                                       \                                         

#SBATCH --job-name=eval-bwc-ava          # the name of the job                                                                         \                                   
#SBATCH --exclude=node718                                                                                                                                                   
#SBATCH --output=evaloutput.txt            # where stdout and stderr will write to                                                    \                                         

#SBATCH -A visualai                                                                                                               \                                         

#SBATCH --gres=gpu:1                   # number of GPUs your job requests                                                         \                                         

#SBATCH --mem=32G                      # amount of memory needed                                                                  \                                         

#SBATCH --time=24:00:00                 # limit on total runtime                                                                 \                                           


source ~/.bashrc
conda activate pytorch-openpose
cd /n/fs/vai-bwc-scr/action_detection/scripts
python eval_bwc_annos.py
