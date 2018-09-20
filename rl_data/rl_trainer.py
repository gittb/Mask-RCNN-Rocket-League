"""
Mask R-CNN Multi class implmentation on a dataset of Rocket League images to find cars, ball, large boost pads, and differnete between the two goals.

Modification of the Color splash example to accept multiple classes and predict on rocket league images dataset.
Modifed by Ben Gitter

Original Author information:
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla


------------------------------------------------------------

Usage: run from the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 RL.py train --dataset=/path/to/RL/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 RL.py train --dataset=/path/to/RL/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 RL.py train --dataset=/path/to/RL/dataset --weights=imagenet
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

#scimage SHUT UP
import warnings
warnings.filterwarnings("ignore") #suppresses all warning from python


# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
# Configurations
############################################################

#mod
class_map = {
    'car' : 1,
    'ball' : 2,
    'boost' : 3,
    'orange_goal' : 4,
    'blue_goal' : 5
}



class RLConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "RL"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # Background + RL #mod

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 80

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.8


############################################################
#  Dataset
############################################################

class RLDataset(utils.Dataset):

    def load_RL(self, dataset_dir, subset):
        """Load a subset of the RL dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # adding our classes into the dataset for reference when predicting later
        self.add_class("RL", 1, "car")
        self.add_class("RL", 2, "ball")
        self.add_class("RL", 3, "boost")
        self.add_class("RL", 4, "orange_goal")
        self.add_class("RL", 5, "blue_goal")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {type of object : object name},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }


        # we want the X and Y cords of the polygon and then also the class idents
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys because the outer most are all the same

        # Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            classes = [r['region_attributes'] for r in a['regions'].values()] #modification needed to accept classes

            # loads our image name and creates a path, then reads the image for its dims
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            #mod
            self.add_image(
                "RL",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                classes=classes) # added to arguments to include feeding of classes

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a RL dataset image, delegate to parent class.
        image_info = self.image_info[image_id]

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_ids = [] #array of classes
        for i, p in enumerate(info["polygons"]):
            class_name = info['classes'][i]['Objects']
            class_ids.append(class_map[class_name]) #checking class against the class map and adding it to the classes list

            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        #rewritten to accept multiple classes
        return mask.astype(np.bool), np.array(class_ids, dtype=np.int32)  #sends a bool map of classes and the list of maps

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "RL":
            return info["path"]
        else:
            print('Image Not from RL dataset')



def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = RLDataset()
    dataset_train.load_RL(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = RLDataset()
    dataset_val.load_RL(args.dataset, "val")
    dataset_val.prepare()


    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=15,
                layers='heads')


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect RLs.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/RL/dataset/",
                        help='Directory of the RL dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = RLConfig()
    else:
        class InferenceConfig(RLConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        print('invalid args')
        exit()

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
