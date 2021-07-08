import pickle as pkl
import torch
import os
import mmcv
import sys
import cv2
import copy
import numpy as np
import argparse
import subprocess
import math
from PIL import Image, ImageDraw

root_dir = '/n/fs/vai-bwc-scr/'

TW_dir     = os.path.join(root_dir, 'annotations/TW_annotations')
TW_paths   = [os.path.join(TW_dir, x) for x in os.listdir(TW_dir)]
title_path = os.path.join(root_dir, 'annotations/URLs/ids_and_titles.txt')

openpose_path = os.path.join(root_dir, 'pytorch-openpose/')
sys.path.append(openpose_path)
from src import util

track_path = os.path.join(root_dir, 'demo/sort/')
sys.path.append(track_path)
from sort import *

parser = argparse.ArgumentParser()
parser.add_argument('--chunk_num', type=int, help='which chunk of videos to process')
args = parser.parse_args()

chunk_dir = os.path.join(root_dir, 'methods/vid_name_chunks')
vid_dir = os.path.join(root_dir, 'videos')

person_dir = os.path.join(root_dir, 'outputs/annotations/person_detection')
face_dir = os.path.join(root_dir, 'outputs/annotations/face_detection')
pose_dir = os.path.join(root_dir, 'pose_detection/detections')

out_dir = os.path.join(root_dir, 'demo/TW_clips')


#with open(os.path.join(chunk_dir, 'chunk_{}.txt'.format(args.chunk_num))) as f:
#    for line in f:
#        vid_names.append(line.strip())
#anno_dir = os.path.join(root_dir, 'annotations/categories')
#anno_file = os.path.join(anno_dir, 'category_annotations.txt') 
#with open(os.path.join(anno_file)) as f:                                                    
#    for line in f:                                                                                                                
#        vid_names.append(line.strip())
        
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
                timeframe = []
                for t in [t1, t2]:
                    minute, second = int(t[:2]), int(t[-2:])
                    timeframe.append(minute*60 + second)
                TWs[curr_ID].append(timeframe)
                
ids = list(IDs_to_titles.keys())
titles = list(IDs_to_titles.values())

top20TWs = list(sorted(TWs, key=lambda key:len(TWs[key]), reverse=True))[0:min(20,len(TWs))]
vid_names = [IDs_to_titles[vid_id] for vid_id in top20TWs]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

img_size = 416

def draw_bbox_cv2(frame, tracked_objects):
    frame_w, frame_h, _ = frame.shape
    pad_x = max(frame_w - frame_h, 0) * (img_size / max(frame.shape))
    pad_y = max(frame_h - frame_w, 0) * (img_size / max(frame.shape))
    unpad_h = img_size  - pad_y
    unpad_w = img_size  - pad_x
    
    txt = 'person'
            
    for x1, y1, x2, y2, obj_id in tracked_objects:
        #if not cls == 0.0:
        #    continue
        box_h, box_w = ((y2 - y1) / unpad_h) * frame_w, ((x2 - x1) / unpad_w) * frame_h
        y1, x1 = int(((y1 - pad_y // 2) / unpad_h) * frame_w), int(((x1 - pad_x // 2) / unpad_w) * frame_h)
        y2, x2 = int(y1 + box_h), int(x1 + box_w)
        start = (x1,y1)
        end = (x2,y2)
        color = (0,255,0)
        #cv2.rectangle(frame, start, end, color, 1)
        idStart = (x1, y1-35)
        idEnd = (x1+len(txt)*19+60,y1)
        #cv2.rectangle(frame, idStart, idEnd, color, -1)
        #cv2.putText(frame, txt + "-" + str(int(obj_id)), 
        #    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
        #    1, (255,255,255), 1)
    return frame

def draw(frame, person_detections, face_detections, pose_detections):
    canvas = copy.deepcopy(frame)
    if pose_detections is not None:
        (candidate, subset) = pose_detections
        canvas = util.draw_bodypose(canvas, candidate, subset)
    if person_detections is not None:
        canvas = draw_bbox_cv2(canvas, person_detections)
    if face_detections is not None:
        for x1, y1, x2, y2, obj_id in face_detections:
            #face_box = box.tolist()
            #start    = (int(face_box[0]), int(face_box[1]))
            #end      = (int(face_box[2]), int(face_box[3]))
            cls = 'face'
            x1 = int(x1)
            y1 = int(y1)
            start    = (x1,y1)
            end      = (int(x2), int(y2))
            color    = (255,0,0)
            cv2.rectangle(canvas, start, end, color, 1)
            idStart = (x1,y1-35)
            idEnd = (x1+len(cls)*19+60,y1)
            #cv2.rectangle(canvas, idStart, idEnd, color, -1)
            #cv2.putText(canvas, cls + "-" + str(int(obj_id)),
             #       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
              #      1, (255,255,255), 1)

    return canvas

def draw_vid(vid_name, vid_dir, person_dir, face_dir, pose_dir, out_dir, blur):
    detect_name = vid_name[:-4].replace(' ', '_') + '.pkl'
    out_name = vid_name[:-4].replace(' ', '_') + '_TWs_overlay.mp4'
    tmp_name = vid_name[:-4].replace(' ', '_') + '_tmp.mp4'
    tmp_path = os.path.join(out_dir, tmp_name)
    out_path = os.path.join(out_dir, out_name)
    print('Processing video {}'.format(vid_name))
    sys.stdout.flush()
    if os.path.exists(out_path):
        print('\tVideo already processed. Skipping.')
        return
    
    subprocess.call(["python",os.path.join(root_dir,"action_detection/slowfast/tools/run_net.py"),"--cfg",str(os.path.join(root_dir,"action_detection/slowfast/configs/AVA/c2/SLOWFAST_32x2_R101_50_50_v2.1.yaml")),"TEST.ENABLE","False", "DEMO.ENABLE", "True", "DEMO.INPUT_VIDEO",str(os.path.join(vid_dir, vid_name)),"DEMO.OUTPUT_FILE",str(tmp_path),"OUTPUT_DIR","./"])

    mot_tracker = Sort()
    face_tracker = Sort()
    person_dict = pkl.load(open(os.path.join(person_dir, detect_name), 'rb'))
    face_dict = pkl.load(open(os.path.join(face_dir, detect_name), 'rb'))
    pose_dict = pkl.load(open(os.path.join(pose_dir, detect_name), 'rb'))

    video = mmcv.VideoReader(tmp_path)
    fps = round(video.fps)
    dim = (video.width, video.height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_tracked = cv2.VideoWriter(out_path, fourcc, fps, dim)
    vid_id = ids[titles.index(vid_name)]
    
    pose_detections = []
    frames = []
    tracked_objects = []
    tracked_faces = []
    for frame_id in range(len(video)):
        #print('Frame {}'.format(frame_id+1), end='\r')
        frame = video[frame_id]

        try:
            f = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except:
            print('Frame {} is faulty. Ending write.'.format(frame_id+1))
            sys.stdout.flush()
            break 

        if blur:
            for timestamp in TWs[vid_id]:
                if frame_id/fps >= timestamp[0] and frame_id/fps <= timestamp[1]:
                    frame = cv2.blur(frame, (255,255))
        frames.append(frame)
        
        person_detection = person_dict[frame_id] if frame_id in person_dict else None
        if person_detection is not None:
            person_detection = np.array([person for person in person_detection if person[-1] == 0.0])

        face_detection = face_dict[frame_id] if frame_id in face_dict else None
        
        pose_detections.append(pose_dict[frame_id] if frame_id in pose_dict else None)
        
        tracked_objects.append(mot_tracker.update(person_detection) if person_detection is not None else None) 
        tracked_faces.append(face_tracker.update(face_detection) if face_detection is not None else None)
        
        if frame_id % 2 != 0:
            persons = []
            faces = []
            if tracked_objects[0] is None or tracked_objects[1] is None:
                persons = None
            else:
                persons = [obj for obj in tracked_objects[1] if obj[-1] in tracked_objects[0][:,-1]]
            
            if tracked_faces[0] is None or tracked_faces[1] is None:
                faces = None
            else:
                faces = [obj for obj in tracked_faces[1] if obj[-1] in tracked_faces[0][:,-1]]
            
            for i in range(len(frames)):
                frame_draw = draw(frames[i], persons, faces, pose_detections[i])  
                video_tracked.write(np.array(frame_draw))

            pose_detections = []
            frames = []
            tracked_objects = []
            tracked_faces = []
        
        if frame_id >= 900 * fps:
            break
            
    video_tracked.release()
    
    full_overlay = mmcv.VideoReader(out_path)
    for timestamp in TWs[vid_id]:
        start_sec = max(timestamp[0]-5,0)
        end_sec = min(timestamp[1]+5,math.floor((len(full_overlay)-1)/fps))
        clip_path = os.path.join(out_dir, 'clips/'+vid_name[:-4].replace(' ', '_') + '_TW_'+'%02d:%02d'%(start_sec//60,start_sec%60)+'-%02d:%02d'%(end_sec//60,end_sec%60)+'.mp4') 
        clip_tracked = cv2.VideoWriter(clip_path, fourcc, fps, dim)
        
        for frame_id in range(start_sec*fps, end_sec*fps):
            clip_tracked.write(full_overlay[frame_id])
        
        clip_tracked.release()

    os.remove(tmp_path)
    #os.remove(out_path)


if __name__ == "__main__":
    print('Processing {} videos.'.format(len(vid_names)), flush=True)
    for vid_name in vid_names:
       # vid_info = vid_name.split('\t')
       # name = vid_info[0]
        #cat = vid_info[1]
        #out = os.path.join(out_dir, cat)
        blur = False
        #vid_id = ids[titles.index(name)]
        #if vid_id in TWs:
        #    blur = True
        draw_vid(vid_name, vid_dir, person_dir, face_dir, pose_dir, out_dir, blur)
