"""
This module calculates the statistics of the dataset,
including the mean and std of (R,G,B) channel.


A pytorch bug is found here.
torch.mean() won't work correctly for very big tensor.
"""

import os
import torch
import torchvision.transforms
import json
from PIL import Image
import numpy as np

# partition_path = '../../data/internal/intermediate_1/partitions/split_both_hands.json'
# img_path = '../../data/internal/intermediate_1/images'

partition_path = '../../data/external/cmu_panoptic_hands/intermediate_1/partitions/partition.json'
img_path = '../../data/external/cmu_panoptic_hands/intermediate_1/imgs'
with open(partition_path, 'r') as f:
	partition = json.load(f)
train_records = partition['train']
print(len(train_records))


img_num = len(train_records)
all_images = torch.zeros(3, img_num, 256, 256)


# idx = 0
# for record in train_records:
# 	img = Image.open(os.path.join(img_path, record[0], record[1]))
# 	all_images[:,idx, ...] = torchvision.transforms.ToTensor()(img)
# 	idx = idx+1

# print(np.mean(all_images.reshape(3, len(train_records)*256*256), axis=1))
# print(np.std(all_images.reshape(3, len(train_records)*256*256), axis=1))

# stop


# print(all_images.shape)
# all_images = all_images.view(3, -1)
# print(all_images.shape)
# print(all_images.mean(1))
# print(torch.mean(all_images, dim=1))


# print(all_images[0].mean())
# print(torch.sum(all_images[0])/1000/256/256)
# print('--------numpy---------')
# print(np.mean(all_images.numpy(), axis=1))
# stop

# print(all_images.std(1))


ave = torch.zeros(3)
for record in train_records:
	# img = Image.open(os.path.join(img_path, record[0], record[1]))
	img = Image.open(os.path.join(img_path, record))
	ave = ave + torchvision.transforms.ToTensor()(img).view(3,-1).mean(1)
ave = ave/len(train_records)
print(ave)

dev = torch.zeros(3)
for record in train_records:
	# img = Image.open(os.path.join(img_path, record[0], record[1]))
	img = Image.open(os.path.join(img_path, record))

	# print((torchvision.transforms.ToTensor()(img).view(3,-1).shape))
	# print(torchvision.transforms.ToTensor()(img).view(3,-1) - ave.unsqueeze(1))
	dev = dev + torch.sum((torchvision.transforms.ToTensor()(img).view(3,-1) - ave.unsqueeze(1))**2, dim=1)
dev = dev/len(train_records)/256/256
std = dev**(1/2)
print(std)

