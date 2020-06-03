import json
import os
import numpy as np
from utils import utils

def get_pck_line(gdtt_label_file, pred_label_file, pck_sigmas):

    pcks = np.zeros(len(pck_sigmas))
    for image_name in pred_label_dict.keys():
        pred_label = pred_label_dict[image_name]['pred_label']
        pred_resol = pred_label_dict[image_name]['resol']
        gdtt_label = gdtt_all_label_dict[image_name]
        for sigma_id in range(len(pck_sigmas)):
            sigma = pck_sigmas[sigma_id]
            pck = utils.PCK(pred_label, gdtt_label, pred_resol, sigma=sigma)
            pcks[sigma_id]= pcks[sigma_id] + pck
    ave_pcks = pcks/len(pred_label_dict)
    return ave_pcks


pck_sigmas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
# get gdtt labels
gdtt_label_file = '../data/external/cmu_panoptic_hands/intermediate_1/labels.json'
with open(gdtt_label_file, 'r') as f:
    gdtt_all_label_dict = json.load(f)
# get pred_labels
pred_label_file = '../checkpoint/OpenPoseHand/03-31-16-57-25/valid_pred_labels/epoch_121.json'
with open(pred_label_file, 'r') as f:
    pred_label_dict = json.load(f)
pcks_1 = get_pck_line(gdtt_label_file, pred_label_file, pck_sigmas)
print(pcks_1)


pred_label_file = '../checkpoint/OpenPoseHand/03-31-16-57-25/valid_gm_pred_labels/gdtt_gm_labels.json'
with open(pred_label_file, 'r') as f:
    pred_label_dict = json.load(f)
pcks_2 = get_pck_line(gdtt_label_file, pred_label_file, pck_sigmas)
print(pcks_2)
