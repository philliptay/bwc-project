import argparse
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_file', type=str, default='/n/fs/vai-bwc-scr/action_detection/annotations/audit.json')
    args = parser.parse_args()
    with open(args.anno_file) as f:
        anno_file = json.load(f)
    print('Video 1:')
    vid1 = anno_file[0]
    #print(vid1)
    print('Num frames: ',len(vid1['data']['image_url']))
    print('Video annotation data: ')
    
    print(vid1['result'].keys())
    print('bbox 1:')
    
    for bbox in vid1['result']['data']:
        for key, val in bbox.items():
            print('Key: ',key)
            print('Val: ',val)

    #for info in vid1['result']['info']:
     #   print(info['header'])
      #  print(info['value'])
        
if __name__ == '__main__':
    main()   
