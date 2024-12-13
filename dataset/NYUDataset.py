import json
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from dataset.BaseDataset import BaseDataset
from dataset.utils import get_hdf5_array
import cv2


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
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image / 255.0

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        # depth = np.clip(depth, 0.0, self.depth_threshold)

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
    new_path = image_path.replace("image", "depth")
    return new_path


def get_meta(data_root, meta_json):
    idx = 0
    for root, dir, files in os.walk(data_root):
        if dir != "image":
            continue
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
            print("Saving to meta data:", entry)
            with open(meta_json, 'a') as f:
                json.dump(entry, f)
                f.write('\n')


if __name__ == "__main__":
    data_root = "/dataset/vfayezzhang/dataset/sunrgbd/SUNRGBD/kv1/NYUdata/"

    meta_json = '/dataset/vfaezzhang/dataset/sunrgbd/SUNRGBD/kv1/NYUdata/meta_data.json'

    if not os.path.exists(meta_json):
        get_meta(data_root=data_root, meta_json=meta_json)

    dataset = SintelDataset()
    print(f"Dataset length: {len(dataset)}")

    for id, (image, depth) in enumerate(dataset):
        print(
            f"Id: {id}, Image shape: {image.shape}, Depth shape: {depth.shape}, Image range: {image.min()} - {image.max()}, Depth range: {depth.min()} - {depth.max()}")
        print(f"If depth has nan: {torch.isnan(depth).any()}")
        print(f"If depth has inf: {torch.isinf(depth).any()}")
