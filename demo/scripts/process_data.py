import pickle as pkl
import json
import os
import mmcv
import sys
import cv2 
import numpy as np
from PIL import Image


root_dir = '/n/fs/vai-bwc-scr/'
vid_dir = os.path.join(root_dir, 'videos')
title_path = os.path.join(root_dir, 'annotations/URLs/ids_and_titles.txt')
detections_dir = os.path.join(root_dir, 'outputs/annotations/person_detection/')
out_dir = os.path.join(root_dir, 'demo/sample')

anno_path = os.path.join(root_dir, 'demo/annotations/anno.txt')
IDs_to_titles = {}
with open(title_path, 'r') as f:
    for line in f:
        line = line.strip()
        ID, title = line[:9], line[10:]
        title = title.replace(' ', '*')
        title = ''.join(title.split())
        title = title.replace('*', ' ') + '.mp4'
        IDs_to_titles[ID] = title
tbs = {}
with open(anno_path, 'r') as f:
    for line in f:
        if 'vimeo' in line: 
            curr_ID = line.strip().split()[0][-9:]
            tbs[curr_ID] = []
        elif line.strip() == '':
            continue
        else:
            line_split = line.strip().split()
            t1, t2 = line_split[:2]
            timeframe = [0, 900]         
            for i, t in enumerate([t1, t2]):
                if t == '-':
                    continue
                minute, second = int(t[:2]), int(t[-2:])
                timeframe[i] = minute*60 + second
            tbs[curr_ID].append(timeframe)
            
            
ids = list(IDs_to_titles.keys())
titles = list(IDs_to_titles.values())
vid_names = [IDs_to_titles[vid_id] for vid_id in tbs]

img_size = 416

def scale_write_boxes(frame, frame_id, detections, json_output):
    frame_h, frame_w = frame.size
    pad_x = max(frame_w - frame_h, 0) * (img_size / max(frame.size))
    pad_y = max(frame_h - frame_w, 0) * (img_size / max(frame.size))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is not None:
        # scale boxes to correct frame display size
        for x1, y1, x2, y2, _, _, cls in detections:
            if not cls == 0.0:
                continue
            box_h, box_w = ((y2 - y1) / unpad_h) * frame_w, ((x2 - x1) / unpad_w) * frame_h
            y1, x1 = ((y1 - pad_y // 2) / unpad_h) * frame_w, ((x1 - pad_x // 2) / unpad_w) * frame_h
            y2, x2 = y1 + box_h, x1 + box_w
            # add bbox with info to json output dict
            json_output['result']['data'].append(
                {'type'  : 'rect', # label type: rectangular box
                 'label' : [], # label corresponding to BasicAI project template 
                 'code'  : [], # explanation of label
                 'category' : [], # class of label
                 'coordinate' : [ # four points clockwise from top left
                     {'x': x1, 'y': y1},
                     {'x': x2, 'y': y1},
                     {'x': x2, 'y': y2},
                     {'x': x1, 'y': y2}
                 ],
                 'editable' : True, # allow moodification
                 'frame' : frame_id # frame that bbox corresponds to  
                }
            )
            
  
def output_data(vid_name):
    name = vid_name[:-4].replace(' ', '_')
    out_vid_dir = os.path.join(out_dir, name)
    if not os.path.exists(out_vid_dir):
        os.mkdir(out_vid_dir)
    out_json_dir = os.path.join(out_vid_dir, 'ai_result')
    if not os.path.exists(out_json_dir):
        os.mkdir(out_json_dir)
    out_json_path = os.path.join(out_json_dir, name+'.json')
    detect_name = name + '.pkl'
    detect_dict = pkl.load(open(os.path.join(detections_dir, detect_name), 'rb'))
   
    video = mmcv.VideoReader(os.path.join(vid_dir, vid_name))
    fps = round(video.fps)
    dim = (video.width, video.height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    vid_id = ids[titles.index(vid_name)]
    tb = []
    if not tbs[vid_id]:
        tb = [[0, len(video)]]
    else :
        tb = tbs[vid_id]
   
    json_output = {'data': {'image_url' : []}, 'result': {'data' : []}}
    for t in tb:
        for frame_id in range(t[0]*fps, min(t[1]*fps, len(video))):
            # extract only 1 frame / sec
            if frame_id % fps != 0:
                continue

            frame = video[frame_id]
            try:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            except:
                print('Frame {} is faulty. Ending write.'.format(frame_id+1))
                sys.stdout.flush()
                break

            # write out frame as image
            out_img_dir = os.path.join(out_vid_dir, 'images')
            if not os.path.exists(out_img_dir):
                os.mkdir(out_img_dir)
            img_name = name+'_'+str(frame_id//fps)+'.jpg'
            frame.save(os.path.join(out_img_dir, img_name))
            json_output['data']['image_url'].append(img_name)

            # get all person detections for frame
            detections = detect_dict[frame_id] if frame_id in detect_dict else None

            # scale and write boxes to json output
            frame_draw = scale_write_boxes(frame, frame_id//fps, detections, json_output)

    with open(out_json_path, 'w') as f:
        json.dump(json_output, f)



  
   

if __name__ == "__main__":
    print('Processing {} videos.'.format(len(vid_names)), flush=True)
    for i, vid_name in enumerate(vid_names):
        output_data(vid_name)
        if i == 15:
            break
     
