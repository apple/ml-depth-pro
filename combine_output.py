import json
import os

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch
import src.depth_pro.depth_pro as depth_pro
from dataset.HypersimDataset import HypersimDataset
from dataset.SintelDataset import SintelDataset
from dataset.utils import get_hdf5_array


def get_dataset(dataset_name):
    if dataset_name == "Hypersim":
        return HypersimDataset()
    elif dataset_name == "Sintel":
        return SintelDataset()


if __name__ == "__main__":
    dataset_name = "Sintel"
    res_root = r'/dataset/vfayezzhang/test/depth-pro/infer/vis/'
    save_root = os.path.join(res_root, 'combine', dataset_name)
    os.makedirs(res_root, exist_ok=True)
    os.makedirs(save_root, exist_ok=True)
    dataset = get_dataset(dataset_name)
    model_names = ['marigold', 'dav2', 'depth-pro']

    for id, (image, depth_gt) in enumerate(dataset):
        image_numpy = image.cpu().numpy().transpose(1, 2, 0)
        depth_gt_np = depth_gt.squeeze().cpu().numpy()

        # 图片列表
        images = [image_numpy, depth_gt_np]
        titles = ["Origin Image", "Ground Truth"]
        for _model_name in model_names:
            predict_depth_path = os.path.join(res_root, _model_name, dataset_name, f"{id}.png")
            predict_depth = cv2.imread(predict_depth_path, cv2.IMREAD_GRAYSCALE)
            predict_depth = predict_depth.astype(np.float32) / 255.0
            images.append(predict_depth)
            titles.append(_model_name)
        flag = 0
        for image in images:
            if image is None:
                flag = 1
        if flag == 1:
            continue
        # 创建一个水平拼接的图
        fig, axes = plt.subplots(1, len(images), figsize=(30, 10))

        for i, ax in enumerate(axes):
            ax.imshow(images[i])
            ax.set_title(titles[i])
            ax.axis("off")

        # 调整布局
        plt.tight_layout()

        plt.savefig(os.path.join(save_root, f"{id}.png"))
