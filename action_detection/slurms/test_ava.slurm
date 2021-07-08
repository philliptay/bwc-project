#!/bin/bash                                                                                                                       \
                                                                                                                               
#SBATCH --job-name=test-slowfast          # the name of the job                                                                         \
#SBATCH --exclude=node718                                                                                                                                   
#SBATCH --output=output.txt            # where stdout and stderr will write to                                                    \
                                                                                                                                   
#SBATCH -A visualai                                                                                                               \
                                                                                                                                   
#SBATCH --gres=gpu:1                   # number of GPUs your job requests                                                         \
                                                                                                                                   
#SBATCH --mem=32G                      # amount of memory needed                                                                  \
                                                                                                                                   
#SBATCH --time=4:00:00                 # limit on total runtime                                                                 \
                                                                                                                                  

source ~/.bashrc
conda activate pytorch-openpose
cd /n/fs/vai-bwc-scr/action_detection/slowfast
python tools/run_net.py \
  --cfg configs/AVA/c2/SLOWFAST_64x2_R101_50_50.yaml \
  DATA.PATH_TO_DATA_DIR /n/fs/vai-bwc-scr/action_detection/ava/ \
  TEST.CHECKPOINT_FILE_PATH /n/fs/vai-bwc-scr/action_detection/slowfast/demo/AVA/SLOWFAST_64x2_R101_50_50.pkl \
  TEST.CHECKPOINT_TYPE caffe2 \
  TRAIN.ENABLE False \
#  DEMO.PREDS_BOXES /n/fs/vai-bwc-scr/action_detection/annotations/BC1_frames_shifted.csv    

#python tools/run_net.py --cfg configs/AVA/c2/SLOWFAST_32x2_R101_50_50_v2.1.yaml TEST.ENABLE True
