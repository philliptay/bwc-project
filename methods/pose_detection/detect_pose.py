import pickle as pkl
import torch
import os
import mmcv
import sys
import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image, ImageDraw

root_dir = '/n/fs/vai-bwc-scr/'
openpose_path = os.path.join(root_dir, 'pytorch-openpose/')
sys.path.append(openpose_path)
from src import model
from src import util
from src.body import Body

body_estimation = Body(os.path.join(openpose_path,'model/body_pose_model.pth'))

parser = argparse.ArgumentParser()
parser.add_argument('--chunk_num', type=int, help='which chunk of videos to process')
args = parser.parse_args()

chunk_dir = os.path.join(root_dir, 'methods/vid_name_chunks')
vid_dir = os.path.join(root_dir, 'videos')
out_dir = os.path.join(root_dir, 'pose_detection/detections')
vid_names = []

with open(os.path.join(chunk_dir, 'chunk_{}.txt'.format(args.chunk_num))) as f:
    for line in f:
        vid_names.append(line.strip())

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

def get_vidcap(filepath):
    return cv2.VideoCapture(filepath)

def detect_pose(imgs):
    detections = []
    for img in imgs:
        candidate, subset = body_estimation(img)
        detections.append([candidate, subset])
    return detections

def process_video(vid_name, vid_dir, out_dir):
    save_name = vid_name[:-4].replace(' ', '_') + '.pkl'
    out_path = os.path.join(out_dir, save_name)
    print('Processing video {}'.format(vid_name))
    sys.stdout.flush()
    
    if os.path.exists(out_path):
        print('\tVideo already processed. Skipping.')
        return

    vidcap =  get_vidcap(os.path.join(vid_dir, vid_name))
    fps = round(vidcap.get(5))
    success = True
    frame_ids = []
    imgs = []
    detect_dict = {}
    while success:
        frame_id = vidcap.get(1)
        success, img = vidcap.read()
        if success:
            frame_ids.append(frame_id)
            imgs.append(img)
        # Stop detections if time goes past 15 minutes
        if frame_id >= fps * 900: 
            success = False 
            
        if not success or len(frame_ids) % 16 == 0:
            if len(frame_ids) == 0:
                continue
            detections = detect_pose(imgs)
            for i, detection in enumerate(detections):
                if detection is not None:
                    detect_dict[frame_ids[i]] = detection
            frame_ids = []
            imgs = [] 
            with open(out_path, 'wb') as f:
                pkl.dump(detect_dict, f)  

if __name__ == "__main__":
    print('Processing {} videos.'.format(len(vid_names)), flush=True)
    for vid_name in vid_names:
        process_video(vid_name, vid_dir, out_dir)
