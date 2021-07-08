import os
import shutil

path = '/n/fs/vai-bwc-scr/demo/basic_ai/'

for video_dir in os.listdir(path):
    if not os.path.exists(os.path.join(path, video_dir,'ai_result')):
        continue
    for f in os.listdir(os.path.join(path, video_dir,'ai_result')):
        shutil.move(os.path.join(path, video_dir,'ai_result',f), path)
    for f in os.listdir(os.path.join(path,video_dir,'images')):
        shutil.move(os.path.join(path, video_dir,'images',f), os.path.join(path,video_dir))
    
    os.rmdir(os.path.join(path,video_dir,'ai_result'))
    os.rmdir(os.path.join(path,video_dir,'images'))
