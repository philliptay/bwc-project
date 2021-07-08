from models import *
from utils import *
import argparse
import os, sys, time, datetime, random
import torch
import pickle as pkl
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--chunk_num', type=int, help='which chunk of video to process.')
args = parser.parse_args()

# Set up file pathing
config_path='config/yolov3.cfg'
weights_path='weights/yolov3.weights'
class_path='data/coco.names'

vid_dir = '/n/fs/vai-bwc-scr/videos/'
chunk_dir = '/n/fs/vai-bwc-scr/methods/vid_name_chunks/'
vid_names = []
with open(os.path.join(chunk_dir, 'chunk_{}.txt'.format(args.chunk_num))) as f:
    for line in f:
        vid_names.append(line.strip())
out_dir = '/n/fs/vai-bwc-scr/outputs/annotations/person_detection/'

img_size=416
conf_thres=0.3
nms_thres=0.4

# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_darknet_weights(weights_path)
model.cuda()
model.eval()
classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor
print('Done loading in model.')


# Get bounding-box colors 
#cmap = plt.get_cmap('tab20b') 
#colors = [cmap(i) for i in np.linspace(0, 1, 20)]


def get_vidcap(filepath):
    return cv2.VideoCapture(filepath)

def detect_image(img):
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms=transforms.Compose([transforms.Resize((imh,imw)),
         transforms.Pad((max(int((imh-imw)/2),0), 
              max(int((imw-imh)/2),0), max(int((imh-imw)/2),0),
              max(int((imw-imh)/2),0)), (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 
                        conf_thres, nms_thres)
    return detections[0]

def detect_images(img_list):
    ratio = min(img_size/img_list[0].size[0], img_size/img_list[0].size[1])
    imw = round(img_list[0].size[0] * ratio)
    imh = round(img_list[0].size[1] * ratio)
    img_transforms=transforms.Compose([transforms.Resize((imh,imw)),
         transforms.Pad((max(int((imh-imw)/2),0), 
              max(int((imw-imh)/2),0), max(int((imh-imw)/2),0),
              max(int((imw-imh)/2),0)), (128,128,128)),
         transforms.ToTensor(),
         ])
    img_tensors = []
    for img in img_list:
        img_tensor = img_transforms(img).float()
        img_tensor = img_tensor.unsqueeze_(0)
        img_tensors.append(img_tensor)
    img_tensors = torch.cat(img_tensors, 0)
    input_img = Variable(img_tensors.type(Tensor))
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections,
                        conf_thres, nms_thres)
    return detections

def filter_person_detections(detections):
    person_detects = []
    for detection in detections:
        if detection[-1] == 0.0:
            person_detects.append(detection)
    if len(person_detects) == 0:
        return None
    return np.array(person_detects)

'''
def draw_bbox(img, detections, out_dir, out_name):
    img = np.array(img)
    plt.figure() 
    fig, ax = plt.subplots(1, figsize=(12,9))
    ax.imshow(img)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape)) 
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape)) 
    unpad_h = img_size - pad_y 
    unpad_w = img_size - pad_x 
    person_detected = False
    if detections is not None:
        bbox_color = colors[0]
        # browse detections and draw bounding boxes
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            if cls_pred != 0.0:
                continue
            person_detected = True
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
            color = bbox_color
            bbox = patches.Rectangle((x1, y1), box_w, box_h,
                linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(bbox)
            plt.text(x1, y1, s=classes[int(cls_pred)],
            color='white', verticalalignment='top',
            bbox={'color': color, 'pad': 0}) 
     
    plt.axis('off') 
    # save image 
    if person_detected:
        outfile = os.path.join(out_dir, out_name)
        plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
        print('\tImg {}: Detected person'.format(out_name))
    else:
        print('\tImg {}: No person detected.'.format(out_name))
    plt.close(fig='all') 
'''

def process_video(vid_dir, vid_name, out_dir):
    save_name = vid_name.replace(' ', '_')[:-4] + '.pkl'
    print('Processing frames for video {}.'.format(vid_name))
    if os.path.exists(os.path.join(out_dir, save_name)):
        print('\tVideo already processed. Skipping.')
        return
    vidcap =  get_vidcap(os.path.join(vid_dir, vid_name))
    fps = round(vidcap.get(5))
    success = True
    frame_ids = []
    imgs = []
    detect_dict = {}
    while success:
        frame_id = vidcap.get(1)
        success, img = vidcap.read()
        if success:
            if frame_id % (fps * 60) == 0:
                print('\tOn minute {}'.format(frame_id // (fps * 60)))
            frame_ids.append(frame_id)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgs.append(img)
        # Stop detections if time goes past 15 minutes
        if frame_id >= fps * 900: 
            success = False 
        # If we are at end of video or have batch ready, process the batch.
        if not success or len(frame_ids) % 16 == 0:
            if len(frame_ids) == 0:
                continue
            detections = detect_images(imgs)
            for i, detection in enumerate(detections):
                if detection is not None:
                    detection = detection.cpu().numpy()
                    person_detections = filter_person_detections(detection)
                    if person_detections is not None:
                        detect_dict[frame_ids[i]] = person_detections
            frame_ids = []
            imgs = [] 
    with open(os.path.join(out_dir, save_name), 'wb') as f:
        pkl.dump(detect_dict, f)    
        

if __name__ == "__main__":
    print('Processing {} videos'.format(len(vid_names)), flush=True)
    for vid_name in vid_names:
        process_video(vid_dir, vid_name, out_dir)
