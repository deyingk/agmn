"""
This module helps to partition the dataset into training, validation and test.
Would generate a dictionary.
"""
import random
import json
import os


def confirm_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_small_partition(dataset_name):

    if dataset_name =='cmu_panoptic':

        partition_save_dir = '/home/deyingk/handpose/data/external/cmu_panoptic_hands/intermediate_1/partitions'
        image_names = [str(i).zfill(8) + '.jpg' for i in range(0,14817)]
        partition_dict = {}
        
        SEED = 1111
        random.seed(SEED)
        random.shuffle(image_names)
        partition_dict['train'] = sorted(image_names[:int(len(image_names)*0.8)])[:100]
        partition_dict['valid'] = sorted(image_names[int(len(image_names)*0.8):int(len(image_names)*0.9)])[:10]
        partition_dict['test'] = sorted(image_names[int(len(image_names)*0.9):])[:10]

        confirm_dir(partition_save_dir)
        with open(os.path.join(partition_save_dir, 'small_partition.json'), 'w') as f:
            json.dump(partition_dict, f)

def main(dataset_name):

    if dataset_name =='cmu_panoptic':

        partition_save_dir = '/home/deyingk/handpose/data/external/cmu_panoptic_hands/intermediate_1/partitions'
        image_names = [str(i).zfill(8) + '.jpg' for i in range(0,14817)]
        partition_dict = {}
        
        SEED = 1111
        random.seed(SEED)
        random.shuffle(image_names)
        partition_dict['train'] = sorted(image_names[:int(len(image_names)*0.8)])
        partition_dict['valid'] = sorted(image_names[int(len(image_names)*0.8):int(len(image_names)*0.9)])
        partition_dict['test'] = sorted(image_names[int(len(image_names)*0.9):])

        confirm_dir(partition_save_dir)
        with open(os.path.join(partition_save_dir, 'partition.json'), 'w') as f:
            json.dump(partition_dict, f)







if __name__ =='__main__':
    # main('cmu_panoptic')
    get_small_partition('cmu_panoptic')