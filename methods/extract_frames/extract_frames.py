import sys
import argparse
import cv2
import os

def extract_frames(vid_path, vid_name, frame_out_path):
    vid_filepath = os.path.join(vid_path, vid_name)
    vidcap = cv2.VideoCapture(vid_filepath)
    fps = round(vidcap.get(5))
    fps_to_write = 10.0
    write_per_frame = fps/fps_to_write
    print('Extracting frames from vid {}. FPS: {}'.format(vid_name, fps))
    print('Writing out every {:.2f} frames'.format(write_per_frame))
    img_prefix = vid_name[:-4]
    img_prefix = img_prefix.replace(' ', '_')
    
    try:
        os.mkdir(os.path.join(frame_out_path, img_prefix))
    except OSError as error:
        print('\t{}. Skipping video'.format(error))
        
    success = True
    count = 1
    next_frame = 0.0
    while success:
        frame_id = vidcap.get(1)
        success, image = vidcap.read()
        if success and frame_id >= next_frame:
            print('\tReading frame at {}:{}, {}'.format(str(count//(60*fps_to_write)).zfill(2), str((count%(60*fps_to_write))//fps_to_write).zfill(2), str(success)))
            sys.stdout.flush()
            cv2.imwrite(os.path.join(frame_out_path, img_prefix, '{}.jpg'.format(count+1)), image)
            if count >= 900*fps_to_write:
                print('\tReaching 15 minute limit. Ending extraction.')
                success = False
            next_frame += write_per_frame
            count += 1 
    vidcap.release()

if __name__ == "__main__":
    vid_path = '/n/fs/vai-bwc-scr/videos'
    frame_out_path = '/n/fs/vai-bwc-scr/frames'
    vid_names = os.listdir(vid_path)
    for vid_name in vid_names:
        extract_frames(vid_path, vid_name, frame_out_path)     
