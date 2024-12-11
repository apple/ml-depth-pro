import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from dataset.BaseDataset import BaseDataset
from dataset.utils import get_hdf5_array


class SintelDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        meta_json = "/dataset/sharedir/research/MPI-Sintel/meta_data.json"
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
        image = np.clip(image, 0.0, 1.0)
        depth = np.clip(depth, 0.0, self.depth_threshold)
        return image, depth

    def __getitem__(self, idx):
        '''
        idx: list,int
        Return:
            image: torch.Tensor
            depth: torch.Tensor
        '''
        if isinstance(idx, list):
            return [self.__getitem__(i) for i in idx]
        image_np, depth_np = self.preproess(self.image_paths[idx], self.depth_paths[idx])
        to_tensor = transforms.ToTensor()
        image = to_tensor(image_np)
        depth = to_tensor(depth_np)
        return image, depth


def convert_image_to_depth_path(image_path):
    # 替换路径中的特定部分
    new_path = image_path.replace(
        "/image/training/final/",
        "/depth/training/depth/"
    )
    # 修改文件扩展名为 .dpt
    new_path = os.path.splitext(new_path)[0] + ".dpt"
    return new_path


def get_meta():
    meta_json = '/dataset/sharedir/research/MPI-Sintel/meta_data.json'
    image_root = "/dataset/sharedir/research/MPI-Sintel/image/training/final/"
    idx = 0
    for root, dir, files in os.walk(image_root):
        for file in files:
            image_path = os.path.join(root, file)
            depth_path = convert_image_to_depth_path(image_path)
            if (not os.path.exists(image_path)) or (not os.path.exists(depth_path)):
                print(f"File not found: {image_path}")
                continue
            entry = {
                "id": idx + 1,
                "img_path": image_path,
                "depth_path": depth_path
            }
            idx += 1
            with open(meta_json, 'a') as f:
                json.dump(entry, f)
                f.write('\n')


if __name__ == "__main__":
    get_meta()
    dataset = SintelDataset()
    print(f"Dataset length: {len(dataset)}")

    for id, (image, depth) in enumerate(dataset):
        print(f"Id: {id}, Image shape: {image.shape}, Depth shape: {depth.shape}")
