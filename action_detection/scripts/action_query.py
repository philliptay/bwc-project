#!/usr/bin/env python3

import sys
import numpy as np
import json
import copy
from tqdm import tqdm
from tabulate import tabulate
from fvcore.common.file_io import PathManager

from slowfast.datasets.ava_helper import parse_bboxes_file


class ActionQuery:

    def __init__(self, filename, class_names_path, interval=5):
        """
        Parse action boxes and labels file.
        Return dict in format boxes[video_name][sec] = preds_list
        """

        boxes, _, _ = parse_bboxes_file(
            ann_filenames=[filename],
            ann_is_gt_box=[False],
            detect_thresh=0.7,
            boxes_sample_rate=1,
        )

        self.class2idx = self._get_classes(class_names_path)
        self.interval = interval
        self.VALID_SECS = range(900,1800)
        self.full_action_map = self._gen_action_map(boxes)

    def search(self, search_action, num_results=10):
        idx = None
        if isinstance(search_action, int):
            idx = search_action
        elif isinstance(search_action, str):
            idx = [val for key, val in self.class2idx.items() if search_action in key][0]
    
        if not idx or idx not in self.full_action_map:
            return None
        
        results = self.full_action_map[idx]
        if not results:
            return None
        
        top_results = copy.deepcopy(sorted(results, key=lambda key:float(key[1]), reverse=True)[0:min(num_results, len(results))])
        top = []
        for res in top_results:
            if float(res[1]) == 0:
                break
            top.append([res[0],res[1],res[2]])

        return top

    def get_classes_map(self):
        return self.class2idx
    
    def _get_classes(self, path):
        try:
            with PathManager.open(path, "r") as f:
                class2idx = json.load(f)
                return class2idx
        except Exception as err:
            print("Fail to load file from {} with error {}".format(path, err))
            return

    def _gen_action_map(self, boxes):
        action_map = {}

        # iterate through all videos
        for video_name in boxes.keys():
            # iterate through all valid secs (15 mins)
            for sec in self.VALID_SECS:
                if sec not in boxes[video_name]:
                    continue

                # get predictions from current sec
                for scores in boxes[video_name][sec]:
                    if scores[1]:
                        for i in range(len(scores[1])):
                            if i not in action_map:
                                action_map[i] = [[sec-900, scores[1][i], video_name]]
                            else:
                                action_map[i].append([sec-900, scores[1][i], video_name])
                            
        # perform nms
        self._action_nms(action_map)

        return action_map

    def _action_nms(self, action_map):
        progress = tqdm(len(action_map), desc='nms')
        for idx, action_list in action_map.items():
            actions = np.array(action_list)
            secs = actions[:,0].astype(np.int)
            confidences = actions[:,1].astype(np.float)
            names = actions[:,2]
            
            suppressed_action_list = []
            while True:
                # get index at max confidence action prediction
                max_idx = np.argmax(confidences) 
                # end if all predictions covered
                if confidences[max_idx] == -1:
                    break
                max_sec = secs[max_idx]
                max_name = names[max_idx]
                # get indices of all predictions within interval to be suppressed
                idxs_to_remove = np.where((abs(secs-max_sec) < self.interval) & (names==max_name))[0]
                # remove non-maximum indices
                confidences[idxs_to_remove] = -1
                # add maximum to action list
                suppressed_action_list.append(action_list[max_idx])
 
            action_map[idx] = suppressed_action_list
            progress.update(1)

                
def main(argv):
    print('generating top actions at interval of %s seconds...'%(argv[0]))
    action_query = ActionQuery('/n/fs/vai-bwc-scr/action_detection/annotations/bwc_detections.csv',
                               '/n/fs/vai-bwc-scr/action_detection/slowfast/demo/AVA/ava_classnames.json', int(argv[0]))
    print('done.')
    while True:
        action = input('Enter an action to search: ')
        results = action_query.search(action)
        if results is None or not results:
            print('No results found.')
            continue
        print('Top 10 results for action \'%s\':'%(action))
        print(tabulate(results, headers=['timestamp','confidence', 'video name']))
        print('\n')

if __name__ == "__main__":
    main(sys.argv[1:])
