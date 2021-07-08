#!/usr/bin/env python3                                                                                                                                                                 

import sys
import numpy as np
import json
import argparse
import copy
from tqdm import tqdm
from collections import defaultdict
from tabulate import tabulate
from fvcore.common.file_io import PathManager
from slowfast.datasets.ava_helper import parse_bboxes_file
from slowfast.datasets.cv2_transform import revert_scaled_boxes
from slowfast.utils.ava_eval_helper import (
    make_image_key,
    read_labelmap,
    read_csv,
    run_evaluation
)

ava_cfg = '/n/fs/vai-bwc-scr/action_detection/slowfast/configs/AVA/c2/SLOWFAST_32x2_R101_50_50_v2.1.yaml'
labelmap = '/n/fs/vai-bwc-scr/action_detection/annotations/bwc_ava_action_labelmap.pbtxt'
# AVA annotations for eval comparison
AVA_DETECTIONS_PATH = '/n/fs/vai-bwc-scr/action_detection/annotations/bwc_frames_shifted_0.csv'
AVA_CLASSNAMES_PATH = '/n/fs/vai-bwc-scr/action_detection/slowfast/demo/AVA/ava_classnames.json'
# Police Body-worn Camera Action annotations 
POBCA_PATH = '/n/fs/vai-bwc-scr/action_detection/annotations/audit.json'

class DataLoader:
    def __init__(self, labelmap, ava_annos, class_names_path, bwc_annos, fps):
        
        
        categories, class_whitelist = read_labelmap(labelmap)
        detections = read_csv(ava_annos, class_whitelist, load_score=True)
        d_boxes, d_labs, d_scores = detections
        
        for key in d_boxes:
            # scale to video size
            d_boxes[key] = revert_scaled_boxes(256, np.array(d_boxes[key]), 360, 640)
        
        exclusions = set()
        with open(bwc_annos) as f:
            bwc_data = json.load(f)
        
        gt_boxes = defaultdict(list)
        gt_labels = defaultdict(list)
        gt_scores = defaultdict(list)
        for vid in bwc_data:
            title = vid['data']['image_url'][0].split('/')[0]
            num_frames = len(vid['data']['image_url'])
            for bbox in vid['result']['data']:
                    
                frame_sec = int(bbox['frame'])
                
                image_key = make_image_key(title, frame_sec)
                # ignore first person labels - irrelevant for AVA
                if image_key not in d_labs or 'First Person POV' in bbox['category']:
                    continue
            
                # prediction failure
               # if image_key not in d_labs:
                #    d_labs[image_key] = [int(item['id']) for item in categories]
                 #   d_scores[image_key] = [0.0]*len(categories)

                box_list = bbox['coordinate']
                box = [box_list[0]['y'], box_list[0]['x'], box_list[2]['y'], box_list[2]['x']]
            
                                
                # search for matching BWC labels in AVA category labels
                for label in bbox['labelAttrs']:
                    if len(label['value']) != 0:
                        for label_val in label['value']:
                            for item in categories:                             
                                if item['name'].split(' ')[0] in label_val.lower() or \
                                   item['name'].split('/')[0] in label_val.lower():
                                    gt_boxes[image_key].append(box)
                                    gt_labels[image_key].append(int(item['id']))
                                    gt_scores[image_key].append(1.0)
                                

        groundtruth = gt_boxes, gt_labels, gt_scores
        run_evaluation(categories, groundtruth, detections, exclusions)
    

            
            
def main():
    fps = 30
    dataloader = DataLoader(labelmap, AVA_DETECTIONS_PATH, AVA_CLASSNAMES_PATH, POBCA_PATH, fps)
    


if __name__ == "__main__":
    main()

