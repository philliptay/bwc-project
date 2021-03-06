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
detections_dir = os.path.join(root_dir, 'outputs/person_detection/detections')
out_dir = os.path.join(root_dir, 'pose_detection/output_videos')
vid_names = []
with open(os.path.join(chunk_dir, 'chunk_{}.txt'.format(args.chunk_num))) as f:
    for line in f:
        vid_names.append(line.strip())

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

def draw_pose(frame, detections):

    canvas = frame
    if detections is not None:
        canvas = copy.deepcopy(frame)
        candidate, subset = body_estimation(canvas)
        canvas = util.draw_bodypose(canvas, candidate, subset)
    return canvas

def draw_vid_pose(vid_name, vid_dir, detections_dir,  out_dir):
    detect_name = vid_name[:-4].replace(' ', '_') + '.pkl'
    out_name = vid_name[:-4].replace(' ', '_') + '_PoD.mp4'
    out_path = os.path.join(out_dir, out_name)
    print('Processing video {}'.format(vid_name))
    sys.stdout.flush()
    if os.path.exists(out_path):
        print('\tVideo already processed. Skipping.')
        return

    video = mmcv.VideoReader(os.path.join(vid_dir, vid_name))
    fps = video.fps
    dim = (video.width, video.height)
    detect_dict = pkl.load(open(os.path.join(detections_dir, detect_name), 'rb'))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_tracked = cv2.VideoWriter(out_path, fourcc, fps, dim)
    for frame_id in range(len(video)):
        #print('Frame {}'.format(frame_id+1), end='\r')
        frame = video[frame_id]
        detections = detect_dict[frame_id] if frame_id in detect_dict else None
        frame_draw = draw_pose(frame, detections)
        #video_tracked.write(cv2.cvtColor(np.array(frame_draw), cv2.COLOR_RGB2BGR))
        video_tracked.write(np.array(frame_draw))
        if frame_id >= 900 * fps:
            break
    video_tracked.release()


if __name__ == "__main__":
  print('Processing {} videos.'.format(len(vid_names)), flush=True)
  for vid_name in vid_names:
    draw_vid_pose(vid_name, vid_dir,detections_dir, out_dir)
