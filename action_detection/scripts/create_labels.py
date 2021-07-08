import os
import sys
import numpy as np
import csv

root_dir = '/n/fs/vai-bwc-scr/'

out_file = os.path.join(root_dir, 'action_detection/annotations/bwc_ava_predictions.csv')
frames_dir =  os.path.join(root_dir, 'action_detection/frames/')
vid_names = []
                                                                                                                                                                        
anno_dir = os.path.join(root_dir, 'annotations/categories')
anno_file = os.path.join(anno_dir, 'category_annotations.txt')
with open(os.path.join(anno_file)) as f:
    for line in f:
        vid_names.append(line.strip())

with open(out_file, 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=' ')
    filewriter.writerow(['original_vido_id', 'video_id', 'frame_id', 'path', 'labels'])
    vid_id = 0
    for name in vid_names:
        vid_info = name.split('\t')
        vid = vid_info[0]
        vid_name = vid[:-4].replace(' ', '_')
        frame_id = 1
        frame = vid_name+'_'+str(frame_id).zfill(6)+'.jpg'
        path = os.path.join(frames_dir,vid_name+'/',frame)
        
        while(os.path.exists(path)):
            #print(path)
            filewriter.writerow([vid_name,str(vid_id),str(frame_id-1),path,'""'])
            frame_id+=1
            frame = vid_name+'_'+str(frame_id).zfill(6)+'.jpg'
            path = os.path.join(frames_dir,vid_name+'/',frame)
        vid_id+=1
