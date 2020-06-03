"""
This module crops hands out from the cmu panoptic dateset (hand143_panopticdb).
Groundtruth label is used and the cropped image size is 2.2 x tightestbox.

Saved in 
...../cmu_panoptic_hands/intermediate_1/imgs/         -------where cropped images are saved            
...../cmu_panoptic_hands/intermediate_1/labels.json   -------the labels on all the cropped images.

"""



from PIL import Image
from os.path import join
import os
import json
import random

def confirm_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

dataset_root = '/home/deyingk/handpose/data/external/cmu_panoptic_hands/'
dataset_raw_root = '/home/deyingk/handpose/data/external/cmu_panoptic_hands/' + 'raw/hand143_panopticdb/'
json_name = dataset_root + 'raw/hand143_panopticdb/hands_v143_14817.json'
with open(json_name) as f:
    ori_json = json.load(f)
all_jsons = ori_json['root']

crop_save_root = '/home/deyingk/handpose/data/external/cmu_panoptic_hands/intermediate_1/'
crop_save_imgs_dir = crop_save_root +'imgs'
confirm_dir(crop_save_imgs_dir)
new_label_json = {}
for entry in all_jsons:
    img_name = entry['img_paths']
    img = Image.open(os.path.join(dataset_raw_root, img_name))
    print(img_name)
    x = [coor[0::3][0] for coor in entry['joint_self']]
    y = [coor[1::3][0] for coor in entry['joint_self']]
    tightest_width = max(x) - min(x)
    tightest_height = max(y) - min(y)
    width = 2.2 * max(tightest_width, tightest_height)
    height = width
    center = [(max(x) + min(x)) / 2, (max(y) + min(y)) / 2]
    area = (center[0] - width/2, center[1] - width/2, center[0] + width/2, center[1] + width/2)
    cropped_img = img.crop(area)
    cropped_img.save(os.path.join(crop_save_imgs_dir, os.path.basename(img_name)))
    label = []
    for i in range(21):
        label.append([x[i] - area[0], y[i] - area[1]])
    new_label_json[os.path.basename(img_name)] = label
with open(os.path.join(crop_save_root, 'labels.json'), 'w') as f:
    json.dump(new_label_json, f)

