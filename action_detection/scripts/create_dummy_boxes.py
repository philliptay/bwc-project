import os
import sys
import numpy as np
import csv

root_dir = '/n/fs/vai-bwc-scr/'

in_file = os.path.join(root_dir, 'action_detection/annotations/bwc_ava_predictions.csv')
out_file = os.path.join(root_dir, 'action_detection/annotations/dummy_boxes_and_labels.csv')

with open(out_file, 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    with open(in_file, 'r') as readfile:
        readfile = csv.reader(readfile)
        first = True
        for line in readfile:
            if first:
                first = False
                continue
            filewriter.writerow([line[0].split(' ')[0],int(line[0].split(' ')[2])+900,0.1,0.1,0.1,0.1,'',0.9])
