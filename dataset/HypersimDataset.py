import json

import torchvision
from joblib import Parallel, delayed
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
        self.illum_paths = []
        self.reflect_paths = []
        self.depth_threshold = 200.
        with open(meta_json, "r", encoding="utf-8") as infile:
            for line in infile:
                entry = json.loads(line)
                img_path = entry["img_path"]
                illum_path = img_path.replace('.color.', '.diffuse_illumination.')
                reflect_path = img_path.replace('.color.', '.diffuse_reflectance.')
                depth_path = entry["depth_path"]
                self.image_paths.append(img_path)
                self.illum_paths.append(illum_path)
                self.reflect_paths.append(reflect_path)
                self.depth_paths.append(depth_path)

    def __len__(self):
        return len(self.image_paths)

    def preproess(self, image_path, depth_path, illum_path=None, reflect_path=None):
        image = get_hdf5_array(image_path)
        depth = get_hdf5_array(depth_path)
        # image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        print('-------------raw image', image.shape, image.max(), image.min())
        image = np.clip(image, 0, 1.0)
        depth = np.clip(depth, 0, self.depth_threshold)
        illum, reflect = None, None
        if illum_path is not None:
            illum = get_hdf5_array(illum_path)
            print(f"Range of illum: {illum.min(), illum.max()}")
            illum = np.clip(illum, 0, 1.0)

        if reflect_path is not None:
            reflect = get_hdf5_array(reflect_path)
            print(f"Range of reflect: {reflect.min(), reflect.max()}")
            reflect = np.clip(reflect, 0, 1.0)

        return image, depth, illum, reflect

    def __getitem__(self, idx):
        '''
        idx: list,int
        Return:
            image: torch.Tensor
            depth: torch.Tensor
        '''
        # if isinstance(idx, list):
        #     return [self.__getitem__(i) for i in idx]
        image_np, depth_np, illum, reflect = self.preproess(self.image_paths[idx], self.depth_paths[idx],
                                                            self.illum_paths[idx], self.reflect_paths[idx])
        to_tensor = transforms.ToTensor()
        image = to_tensor(image_np)
        depth = to_tensor(depth_np)
        if illum is not None:
            illum = to_tensor(illum)
        if reflect is not None:
            reflect = to_tensor(reflect)
        # depth = depth_normalize(depth)
        return image, depth, illum, reflect


def clean():
    def process_id(id):
        image_path = image_paths[id]
        depth_path = depth_paths[id]
        image = get_hdf5_array(image_path)
        # Calculate the invalid percentage
        img_invalid_percentage = (((image > 1).sum() / (image > -0x3ff).sum()) +
                                  (image < 0).sum() / (image < 0x3ff).sum()) * 100
        print(f"Image {id} invalid percentage: {img_invalid_percentage}")
        result = {threshold: "valid" if img_invalid_percentage <= threshold else "invalid"
                  for threshold in thresholds}
        return id, result

    image_paths = []
    depth_paths = []
    with open(meta_json, "r", encoding="utf-8") as infile:
        for line in infile:
            entry = json.loads(line)
            image_paths.append(entry["img_path"])
            depth_paths.append(entry["depth_path"])
    print(f"len of image_paths: {len(image_paths)}\n"
          f"len of depth_paths: {len(depth_paths)}")

    thresholds = [5, 10, 15, 20, 25, 30, 35, 40]
    valid_id = {threshold: [] for threshold in thresholds}
    invalid_id = {threshold: [] for threshold in thresholds}

    results = Parallel(n_jobs=-1)(delayed(process_id)(id) for id in range(len(image_paths)))

    # Organize results
    for id, result in results:
        for threshold, status in result.items():
            if status == "valid":
                valid_id[threshold].append(id)
            else:
                invalid_id[threshold].append(id)

    # Print summary
    for threshold in thresholds:
        print(f"Threshold: {threshold}, valid: {len(valid_id[threshold])}, invalid: {len(invalid_id[threshold])}")

    save_root = '/dataset/sharedir/research/Hypersim/EDA'
    os.makedirs(save_root, exist_ok=True)
    for threshold in thresholds:
        with open(os.path.join(save_root, f"valid_files_{threshold}.json"), "w") as outfile:
            for id in valid_id[threshold]:
                json.dump({'id': id, 'img_path': image_paths[id], 'depth_path': depth_paths[id]}, outfile)
                outfile.write("\n")
        with open(os.path.join(save_root, f"invalid_files_{threshold}.json"), "w") as outfile:
            for id in invalid_id[threshold]:
                json.dump({'id': id, 'img_path': image_paths[id], 'depth_path': depth_paths[id]}, outfile)
                outfile.write("\n")


if __name__ == "__main__":
    # clean()
    # input()
    # import torchvision
    #
    dataset = HypersimDataset()  #
    print(f"Dataset length: {len(dataset)}")
    idx = 0
    for id in range(len(dataset)):
        image, depth, illum, reflect = dataset[id]

        print(f"Id: {idx}, Image shape: {image.shape}, Depth shape: {depth.shape}")
        print(image.max(), image.min(), image.mean())
        print(depth.max(), depth.min(), depth.mean())
        save_root = 'vis/hypersim'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        image_save_path = os.path.join(save_root, f'{idx}_image.png')
        depth_save_path = os.path.join(save_root, f'{idx}_depth.png')
        illum_save_path = os.path.join(save_root, f'{idx}_illum.png')
        reflect_save_path = os.path.join(save_root, f'{idx}_reflect.png')

        torchvision.utils.save_image(image.unsqueeze(0), image_save_path)
        torchvision.utils.save_image(illum.unsqueeze(0), illum_save_path)
        torchvision.utils.save_image(reflect.unsqueeze(0), reflect_save_path)

        max_v, min_v = depth.max(), depth.min()
        torchvision.utils.save_image(((depth - min_v) / (max_v - min_v)).unsqueeze(0), depth_save_path)
        idx = int(input('input idx:'))
