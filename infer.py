import json
import os
import time
import cv2
import numpy as np
import torchvision.transforms
from PIL import Image
from matplotlib import pyplot as plt
import torch
import src.depth_pro.depth_pro as depth_pro
from dataset.HypersimDataset import HypersimDataset
from dataset.NYUDataset import NYUDataset
from dataset.SintelDataset import SintelDataset
from dataset.utils import get_hdf5_array

device = "cuda" if torch.cuda.is_available() else "cpu"
# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model = model.to(device).eval()


def save_single_fig(predict_depth, save_root, id):
    predict_depth = (predict_depth - np.min(predict_depth)) / (np.max(predict_depth) - np.min(predict_depth))
    predict_depth = predict_depth * 255.0
    predict_depth = predict_depth.astype(np.uint8)
    save_path = os.path.join(save_root, f"{id}.png")
    print(f"Saving to {save_path}")
    cv2.imwrite(save_path, predict_depth)


def get_dataset(dataset_name):
    if dataset_name == "Hypersim":
        return HypersimDataset()
    elif dataset_name == "Sintel":
        return SintelDataset()
    elif dataset_name == "NYUv2":
        return NYUDataset()


# dataset_name = "Sintel"
# dataset_name = "Hypersim"
dataset_name = "NYUv2"
dataset = get_dataset(dataset_name)
save_root = os.path.join('./vis/depth-pro-test-large', dataset_name)
os.makedirs(save_root, exist_ok=True)
cnt = 0
elapse_time = 0.0
for id, (image, depth_gt) in enumerate(dataset):
    image, depth_gt = image.to(device), depth_gt.to(device)
    image = image.unsqueeze(0)
    image = torchvision.transforms.Resize((1536, 1536))(image)
    image_numpy = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    # print(f"Image range: {np.min(image_numpy), np.max(image_numpy)}")

    # Run inference.
    consume_time = 0
    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        begin_time = time.time()
        prediction, fov = model(image * 2 - 1, test=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        consume_time = time.time() - begin_time
    elapse_time += consume_time
    cnt += 1
    if cnt % 20 == 0:
        print(f"Avg time for {cnt} images: {elapse_time / cnt}")
    # print(f"prediction shape: {prediction.shape}")
    depth = 1 / prediction
    predict_depth_np = depth.cpu().numpy()
    if cnt == 1002:
        break

    # Normalize the depth maps for visualization
    predict_depth_vis = predict_depth_np
    save_single_fig(predict_depth_vis, save_root, id)
