import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

import sys

sys.path.append('/dataset/hailangwu/zk/project/JDU/ml-depth-pro')

from dataset.BaseDataset import BaseDataset
from dataset.utils import get_hdf5_array
import random
import glob
import cv2
import os

meta_json = "/dataset/sharedir/research/Hypersim/valid_files.json"


def depth_normalize(depth):
    depth_mean, depth_std = torch.mean(depth, dim=(1, 2), keepdim=True), torch.std(depth, dim=(1, 2), keepdim=True)
    normalized_depth = (depth - depth_mean) * (depth_std + 1e-12)
    return normalized_depth


class HypersimDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.meta_json = meta_json
        self.image_paths = []
        self.depth_paths = []
        self.depth_threshold = 200.
        with open(meta_json, "r", encoding="utf-8") as infile:
            for line in infile:
                entry = json.loads(line)
                self.image_paths.append(entry["img_path"])
                self.depth_paths.append(entry["depth_path"])

    def __len__(self):
        return len(self.image_paths)

    def preproess(self, image_path, depth_path):
        image = get_hdf5_array(image_path)
        depth = get_hdf5_array(depth_path)
        # image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        print('-------------raw image', image.shape, image.max(), image.min())
        image = np.clip(image, 0, 1.0)
        depth = np.clip(depth, 0, self.depth_threshold)

        return image, depth

    def __getitem__(self, idx):
        '''
        idx: list,int
        Return:
            image: torch.Tensor
            depth: torch.Tensor
        '''
        # if isinstance(idx, list):
        #     return [self.__getitem__(i) for i in idx]
        image_np, depth_np = self.preproess(self.image_paths[idx], self.depth_paths[idx])
        to_tensor = transforms.ToTensor()
        image = to_tensor(image_np)
        depth = to_tensor(depth_np)
        # depth = depth_normalize(depth)
        return image, depth


if __name__ == "__main__":
    image_paths = []
    depth_paths = []
    with open(meta_json, "r", encoding="utf-8") as infile:
        for line in infile:
            entry = json.loads(line)
            image_paths.append(entry["img_path"])
            depth_paths.append(entry["depth_path"])
    print(f"len of image_paths: {len(image_paths)}"
          f"len of depth_paths: {len(depth_paths)}")
    thresholds = [20, 25, 30, 35, 40]
    valid_cnt = {
        20: 0,
        25: 0,
        30: 0,
        35: 0,
        40: 0,
    }
    invalid_cnt = {
        20: 0,
        25: 0,
        30: 0,
        35: 0,
        40: 0,
    }
    for id in range(len(image_paths)):
        for threshold in thresholds:
            image_path = image_paths[id]
            depth_path = depth_paths[id]
            image = get_hdf5_array(image_path)
            depth = get_hdf5_array(depth_path)
            # count the percentage of values larger than 1
            img_invalid_percentage = (image > 1).sum() / image.size
            if img_invalid_percentage > threshold:
                with open(f'Invalid Image {threshold}.json', 'a') as f:
                    json.dump({
                        'id': id,
                        'img_path': image_path,
                        'depth_path': depth_path,
                    }, f)
                    f.write('\n')
                invalid_cnt[threshold] += 1
            else:
                with open(f'Valid Image {threshold}.json', 'a') as f:
                    json.dump({
                        'id': id,
                        'img_path': image_path,
                        'depth_path': depth_path,
                    }, f)
                    f.write('\n')
                valid_cnt[threshold] += 1
    print(valid_cnt)

#     import torchvision
#
#     dataset = HypersimDataset()  #
#     print(f"Dataset length: {len(dataset)}")
#     idx = 0
#     for id in range(len(dataset)):
#         image, depth = dataset[idx]
#         print(f"Id: {idx}, Image shape: {image.shape}, Depth shape: {depth.shape}")
#         print(image.max(), image.min(), image.mean())
#         print(depth.max(), depth.min(), depth.mean())
#         save_root = 'vis/hypersim'
#         if not os.path.exists(save_root):
#             os.makedirs(save_root)
#         image_save_path = os.path.join(save_root, f'{idx}_image.png')
#         depth_save_path = os.path.join(save_root, f'{idx}_depth.png')
#         torchvision.utils.save_image(image.unsqueeze(0), image_save_path)
#         max_v, min_v = depth.max(), depth.min()
#         torchvision.utils.save_image(((depth - min_v) / (max_v - min_v)).unsqueeze(0), depth_save_path)
#         idx = int(input('input idx:'))
