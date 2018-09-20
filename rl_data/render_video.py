# # Mask R-CNN - Render video from detector
#the oh so holy imports
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find root of library

# Import auxiliary libaries
from backend_vis import *
import mrcnn.model as modellib
from mrcnn.model import log
from mem_top import mem_top
import rl_trainer as RL


config  =  RL.RLConfig()

#################
#CONFIG
#################

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0  select computation device
TEST_MODE = "inference" # Inspect the model in training or inference modes values: 'inference' or 'training'
RL_DIR = 'images' #link to validation dataset
MODEL_DIR = 'logs' #model dir
video_path = 'rl.mp4'
show_boxes = True

#the dict of colors
color_dict = {
    'car' : [.5,0,.8],
    'ball' : [0,1,0],
    'boost' : [1,.5,1],
    'orange_goal' : [0,.5,1],
    'blue_goal' : [1,.6,.2]
}
##############################

config = InferenceConfig()

# Load dataset
dataset = RL.RLDataset()
dataset.load_RL(RL_DIR, "val")

# prep dataset object
dataset.prepare()

print('Classes: ', dataset.class_names)
# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
    weights_path = 'rlcurrent2/mask_rcnn_rl_0015.h5'
    model.load_weights(weights_path, by_name=True)                           

#preping our video modification objects
video_in = cv2.VideoCapture(video_path)
video_out = cv2.VideoWriter('Masked_video.avi', cv2.VideoWriter_fourcc('M','J','P','G'),30,(1200,675))

#detects image and returns
def run_image(img):
    results = model.detect([img], verbose=0) #sending image to model and getting results
    r = results[0] # we only want our main dict of outputs

    #Sends our image to the backend for some processing of model outputs (modified version of RCNN viusalize lib)
    img, _ = render(img, r['rois'], r['masks'], r['class_ids'], 
                        dataset.class_names, r['scores'], 
                        colors=color_dict, show_bbox=show_boxes)
    return img

# SEND IT
print("[#] Annotating... \n")
frame_count = 0
success, frame = video_in.read() #reads in a bool of if there is an images, and the frame as a numpy array
while success:
    video_out.write(run_image(frame)) #runs our detection function and then appends the frame to the video
    frame_count += 1
    print("Frames Rendered: " + str(frame_count), end='\r') #pretty refreshing print statment for those long videos
    success, frame = video_in.read()
    plt.close("all") #matplotlib memory leak? Revisit vis backend at later data. Works for now

video_out.release() #releases and encapsulates the video file
print('[-] Video Released')