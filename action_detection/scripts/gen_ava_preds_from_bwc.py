import pickle as pkl
import torch
import os
import json
import mmcv
import sys
import cv2
import copy
import numpy as np
import argparse
import subprocess
import math

root_dir = '/n/fs/vai-bwc-scr/'
vid_dir = os.path.join(root_dir, 'videos')

bwc_annos_path = '/n/fs/vai-bwc-scr/action_detection/annotations/audit.json'                                                                                                          

with open(bwc_annos_path) as f:                                                                                                                                                       
    bwc_annos = json.load(f)     

titles = []
for vid in bwc_annos:
    title = vid['data']['image_url'][0].split('/')[0]
    if 'AXON' in title:
        title = title[:-4]+' 415.mp4'
    else:
        title = title.replace('_',  ' ')+'.mp4'
    titles.append(title)
        
if __name__ == "__main__":
    for title in titles:
        #if 'Body-Cam Video #1' not in title:
        #    continue
        print('processing video: ' + title)
        subprocess.call(["python",os.path.join(root_dir,"action_detection/slowfast/tools/run_net.py"),"--cfg",
                         str(os.path.join(root_dir,"action_detection/slowfast/configs/AVA/c2/SLOWFAST_32x2_R101_50_50_v2.1.yaml")),
                          "TEST.ENABLE","False", "DEMO.WRITE", "True", "DEMO.ENABLE", "False", "DEMO.INPUT_VIDEO",str(os.path.join(vid_dir, title)),
                          'DEMO.PREDS_BOXES',''])
