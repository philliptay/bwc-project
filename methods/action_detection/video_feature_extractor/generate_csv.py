import os

vid_dir = '/n/fs/vai-bwc-scr/videos/'
vid_names = os.listdir(vid_dir)

feat_dir = '/n/fs/vai-bwc-scr/ht100m_features'

with open('paths.csv', 'w') as f:
    f.write('video_path,feature_path\n')
    for vid_name in vid_names:
        feat_name = vid_name[:-4].replace(' ', '_') + '.npy'
        f.write('{},{}\n'.format(os.path.join(vid_dir, vid_name),
                               os.path.join(feat_dir, feat_name)))

