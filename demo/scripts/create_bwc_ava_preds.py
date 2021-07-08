import pickle as pkl
import torch
import os
import mmcv
import sys
import cv2
import copy
import numpy as np
import argparse
import math


root_dir = '/n/fs/vai-bwc-scr/'

TW_dir     = os.path.join(root_dir, 'annotations/TW_annotations')
TW_paths   = [os.path.join(TW_dir, x) for x in os.listdir(TW_dir)]
title_path = os.path.join(root_dir, 'annotations/URLs/ids_and_titles.txt')

vid_dir = os.path.join(root_dir, 'videos')


IDs_to_titles = {}
with open(title_path, 'r') as f:
    for line in f:
        line = line.strip()
        ID, title = line[:9], line[10:]
        title = title.replace(' ', '*')
        title = ''.join(title.split())
        title = title.replace('*', ' ') + '.mp4'
        IDs_to_titles[ID] = title

TWs = {}
for TW_path in TW_paths:
    if not "BWC" in TW_path or '.swp' in TW_path:
        continue
    with open(TW_path, 'r') as f:
        for line in f:
            if 'vimeo' in line:
                curr_ID = line.strip().split()[0][-9:]
                TWs[curr_ID] = []
            elif line.strip() == '':
                continue
            else:
                line_split = line.strip().split()
                t1, t2 = line_split[:2]
                if len(line_split) > 2:
                    desc = ' '.join(line_split[2:])
                else:
                    desc = 'no description provided'
                timeframe = []
                for t in [t1, t2]:
                    minute, second = int(t[:2]), int(t[-2:])
                    timeframe.append(minute*60 + second)
                TWs[curr_ID].append([timeframe, desc])

ids = list(IDs_to_titles.keys())
titles = list(IDs_to_titles.values())

if __name__ == 'main':
    for title in titles:
        video_name = os.path.join(vid_dir, title)
         subprocess.call(["python",os.path.join(root_dir,"action_detection/slowfast/tools/run_net.py"),"--cfg",str(os.path.join(root_dir,"action_detection/slowfast/configs/AVA/c2/SLOWFAST_32x2_R101_\
50_50_v2.1.yaml")),"TEST.ENABLE","False", "DEMO.ENABLE", "True", "DEMO.INPUT_VIDEO",str(os.path.join(vid_dir, vid_name)),"DEMO.OUTPUT_FILE",str(tmp_path),"OUTPUT_DIR","./])
