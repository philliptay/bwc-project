import os

for subdir, dirs, files in os.walk('/n/fs/vai-bwc-scr/action_detection/action_clips'):
    for filename in files:
        path = subdir + os.sep
        temp = filename.split('@')
        temp1 = temp[1].split('_')
        name = temp[0]
        conf = temp1[0]
        time = temp1[1]
        new_name = conf+'__'+name+'__'+time
        os.rename(path+filename, path+new_name)
