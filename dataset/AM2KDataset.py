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


class AM2KDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        meta_json = '/dataset/vfayezzhang/dataset/AM-2K/validation/validation_meta.json'
        self.meta_json = meta_json
        self.image_paths = []
        with open(meta_json, "r", encoding="utf-8") as infile:
            for line in infile:
                entry = json.loads(line)
                self.image_paths.append(entry["img_path"])

    def __len__(self):
        return len(self.image_paths)

    def preproess(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image / 255.0

        # depth = np.clip(depth, 0.0, self.depth_threshold)

        return image

    def __getitem__(self, idx):
        '''
        idx: list,int
        Return:
            image: torch.Tensor
            depth: torch.Tensor
        '''
        if isinstance(idx, list):
            return [self.__getitem__(i) for i in idx]
        image_np = self.preproess(self.image_paths[idx])
        to_tensor = transforms.ToTensor()
        image = to_tensor(image_np)
        return image


def get_meta(meta_json):
    image_root = "/dataset/vfayezzhang/dataset/AM-2K/validation/original/"
    idx = 0
    image_paths = []
    for root, dir, files in os.walk(image_root):
        for file in files:
            image_path = os.path.join(root, file)
            if (not os.path.exists(image_path)):
                print(f"File not found: {image_path}")
                continue
            image_paths.append(image_path)
    image_paths = sorted(image_paths)
    cnt = 0
    with open(meta_json, 'w') as f:
        for image_path in image_paths:
            cnt += 1
            json.dump({
                'id': cnt,
                'img_path': image_path,
            }, f)
            f.write('\n')


if __name__ == "__main__":
    meta_json = '/dataset/vfayezzhang/dataset/AM-2K/validation/validation_meta.json'

    if not os.path.exists(meta_json):
        get_meta(meta_json=meta_json)

    dataset = AM2KDataset()
    print(f"Dataset length: {len(dataset)}")

    for id, (image, depth) in enumerate(dataset):
        print(f"Id: {id}, Image shape: {image.shape}, Depth shape: {depth.shape}")
