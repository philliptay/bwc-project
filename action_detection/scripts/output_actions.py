import os
import mmcv
import sys
import cv2
import numpy as np
import math
import argparse
from action_query import ActionQuery

root_dir = '/n/fs/vai-bwc-scr/'
vid_dir = os.path.join(root_dir, 'videos/')
out_dir = os.path.join(root_dir, 'action_detection/action_clips/')

def draw_vids(results, action, out_dir):
    for result in results:
        sec = int(result[0])
        confidence = result[1]
        vid_name = result[2]

        vid_path = os.path.join(vid_dir, vid_name+'.mp4')                                                                                                                  
        video = mmcv.VideoReader(vid_path)
        fps = round(video.fps)
        dim = (video.width, video.height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        start_sec = max(sec-5,0)
        end_sec = min(sec+5,math.floor((len(video)-1)/fps))
        action = action.split('/')[0].replace(' ', '_')
        out_path = os.path.join(out_dir, action)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        clip_path = os.path.join(out_path, vid_name.replace(' ', '_') + '_@'+str(confidence)+'_'+'%02d:%02d'%(start_sec//60,start_sec%60)+'-%02d:%02d'%(end_sec//60,end_sec%60)+'.mp4') 
        clip_tracked = cv2.VideoWriter(clip_path, fourcc, fps, dim)
        
        for frame_id in range(start_sec*fps, end_sec*fps):
            clip_tracked.write(video[frame_id])
        
        clip_tracked.release()



if __name__ == "__main__":
    action_query = ActionQuery('/n/fs/vai-bwc-scr/action_detection/annotations/bwc_detections.csv',
                               '/n/fs/vai-bwc-scr/action_detection/slowfast/demo/AVA/ava_classnames.json', 5)
    for action,idx in action_query.get_classes_map().items():
        print('Processing action %s...'%(action), flush=True)
        results = action_query.search(int(idx),50)
        if results is not None:
            draw_vids(results, action, out_dir)
