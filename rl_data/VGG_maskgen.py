import json
import time
import numpy as np
import skimage.draw
import cv2



###################
##################
#Obselete, just used as a scratch pad to better understand processing of json files. Ill include to possibly help others understand
#################
##################


height = 1080
width = 1920


class_map = {
    'car' : 1,
    'ball' : 2,
    'boost' : 3,
    'orange_goal' : 4,
    'blue_goal' : 5
}

#need to create class for photos to be saved with mask

def mask_gen(polygons, classes, height, width, class_map):

    #identifying objects within the polygons
    object_data = []
    for i in range(len(classes)):
        obj_polygon = {
            'x' : polygons[i]['all_points_x'],
            'y' : polygons[i]['all_points_y']
        }
        class_id = classes[i]['Objects']
        
        object_data.append({
            'class' : class_id, 
            'polygon' : obj_polygon
            })

    #drawing polygons and creating masks
    mask = np.zeros([height, width, len(object_data)], dtype=np.uint8)
    class_ids = []
    for num, obj in enumerate(object_data):
        class_ids.append(class_map[obj['class']])
        rr, cc = skimage.draw.polygon(obj['polygon']['y'], obj['polygon']['x'])
        mask[rr, cc, num] = 1
    
    return mask.astype(np.bool), np.array(class_ids, dtype=np.int32)


#loading json
data = json.load(open("via_region_data.json"))

#get all the values from the json
data = list(data.values())

#create new list excluding empty regions
data = [i for i in data if i['regions']]

for annotations in data:
    #getting data from regions
    polygons = [s['shape_attributes'] for s in annotations['regions'].values()]
    classes = [a['region_attributes'] for a in annotations['regions'].values()]
    #masks and array by with size of the number of channels in mask containing the class of each mask
    mask, classes_list = mask_gen(polygons, classes, height, width, class_map)