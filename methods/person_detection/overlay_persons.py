import pickle as pkl
import os
import mmcv
import sys
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('--chunk_num', type=int, help='which chunk of videos to process')
args = parser.parse_args()

root_dir = '/n/fs/vai-bwc-scr/'
chunk_dir = os.path.join(root_dir, 'methods/vid_name_chunks')
vid_dir = os.path.join(root_dir, 'videos')
detections_dir = os.path.join(root_dir, 'outputs/person_detection/detections')
out_dir = os.path.join(root_dir, 'outputs/person_detection/output_videos')
vid_names = []
with open(os.path.join(chunk_dir, 'chunk_{}.txt'.format(args.chunk_num))) as f:
    for line in f:
        vid_names.append(line.strip())

img_size = 416

def draw_bbox_pil(frame, detections):
    frame_h, frame_w = frame.size
    pad_x = max(frame_w - frame_h, 0) * (img_size / max(frame.size))
    pad_y = max(frame_h - frame_w, 0) * (img_size / max(frame.size))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    frame_draw = frame.copy()
    if detections is not None:
        draw = ImageDraw.Draw(frame_draw)
        for x1, y1, x2, y2, _, _, cls in detections:
            if not cls == 0.0:
                continue
            box_h, box_w = ((y2 - y1) / unpad_h) * frame_w, ((x2 - x1) / unpad_w) * frame_h
            y1, x1 = ((y1 - pad_y // 2) / unpad_h) * frame_w, ((x1 - pad_x // 2) / unpad_w) * frame_h
            y2, x2 = y1 + box_h, x1 + box_w
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
    return frame_draw

def draw_vid_bbox(vid_name, vid_dir, detections_dir, out_dir):
    detect_name = vid_name[:-4].replace(' ', '_') + '.pkl'
    out_name = vid_name[:-4].replace(' ', '_') + '_PD.mp4'
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
        frame = video[frame_id]
        try:
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except:
            print('Frame {} is faulty. Ending write.'.format(frame_id+1))
            sys.stdout.flush()
            break 
        detections = detect_dict[frame_id] if frame_id in detect_dict else None
        frame_draw = draw_bbox_pil(frame, detections)
        video_tracked.write(cv2.cvtColor(np.array(frame_draw), cv2.COLOR_RGB2BGR))
        if frame_id >= 900 * fps:
            break
    video_tracked.release()


if __name__ == "__main__":
    print('Processing {} videos.'.format(len(vid_names)), flush=True)
    for vid_name in vid_names:
        draw_vid_bbox(vid_name, vid_dir, detections_dir, out_dir)
