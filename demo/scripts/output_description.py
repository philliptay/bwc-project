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


out_dir = os.path.join(root_dir, 'demo/TW_clips')
 

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

top20TWs = list(sorted(TWs, key=lambda key:len(TWs[key]), reverse=True))[0:min(20,len(TWs))]
vid_names = [IDs_to_titles[vid_id] for vid_id in top20TWs]

out_file = os.path.join(root_dir, 'demo/TW_clips/TW_descriptions.txt')

with open(out_file, 'w') as f:
    for vid_id in top20TWs:
        video = mmcv.VideoReader(os.path.join(vid_dir, IDs_to_titles[vid_id]))
        fps = round(video.fps)
        name =  IDs_to_titles[vid_id][:-4].replace(' ', '_')
        for timestamp, desc in TWs[vid_id]:
            start_sec = max(timestamp[0]-5,0)
            end_sec = min(timestamp[1]+5,math.floor((len(video)-1)/fps))
            line = name+'_TW_'+'%02d:%02d'%(start_sec//60,start_sec%60)+'-%02d:%02d'%(end_sec//60,end_sec%60)+'.mp4: ' + desc
            f.write(line+'\n')
        
