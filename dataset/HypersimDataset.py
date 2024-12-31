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


def depth_normalize(depth):
    depth_mean, depth_std = torch.mean(depth, dim=(1, 2), keepdim=True), torch.std(depth, dim=(1, 2), keepdim=True)
    normalized_depth = (depth - depth_mean) * (depth_std + 1e-12)
    return normalized_depth


class HypersimDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        meta_json = "/dataset/sharedir/research/Hypersim/valid_files.json"
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
        idx = random.randint(0, len(self.image_paths) - 1)
        print(idx)
        image_np, depth_np = self.preproess(self.image_paths[idx], self.depth_paths[idx])
        to_tensor = transforms.ToTensor()
        image = to_tensor(image_np)
        depth = to_tensor(depth_np)
        # depth = depth_normalize(depth)
        return image, depth


if __name__ == "__main__":
    import torchvision

    dataset = HypersimDataset()  #
    print(f"Dataset length: {len(dataset)}")
    for id, (image, depth) in enumerate(dataset):
        print(f"Id: {id}, Image shape: {image.shape}, Depth shape: {depth.shape}")
        print(image.max(), image.min(), image.mean())
        print(depth.max(), depth.min(), depth.mean())
        save_root = 'vis/hypersim'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        image_save_path = os.path.join(save_root, f'{id}_image.png')
        depth_save_path = os.path.join(save_root, f'{id}_depth.png')
        torchvision.utils.save_image(image.unsqueeze(0), image_save_path)
        max_v, min_v = depth.max(), depth.min()
        torchvision.utils.save_image(((depth - min_v) / (max_v - min_v)).unsqueeze(0), depth_save_path)
