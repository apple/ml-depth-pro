import json
import os

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch
import src.depth_pro.depth_pro as depth_pro
from dataset.AM2KDataset import AM2KDataset
from dataset.HypersimDataset import HypersimDataset
from dataset.NYUDataset import NYUDataset
from dataset.SintelDataset import SintelDataset
from dataset.utils import get_hdf5_array


def get_dataset(dataset_name):
    if dataset_name == "Hypersim":
        return HypersimDataset()
    elif dataset_name == "Sintel":
        return SintelDataset()
    elif dataset_name == "NYUv2":
        return NYUDataset()
    elif dataset_name == "AM2K":
        return AM2KDataset()


if __name__ == "__main__":
    import os
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # dataset_name = "Sintel"
    # dataset_names = ['NYUv2']
    dataset_names = ['Sintel']
    res_root = r'/dataset/vfayezzhang/test/depth-pro/infer/vis/'
    for dataset_name in dataset_names:
        save_root = os.path.join(res_root, 'combine', dataset_name)
        os.makedirs(res_root, exist_ok=True)
        os.makedirs(save_root, exist_ok=True)
        dataset = get_dataset(dataset_name)

        model_names = ['dav2-test-large', 'depth-pro-test-large', 'dav2']

        for id, (image, depth_gt) in enumerate(dataset):
            image_numpy = image.cpu().numpy().transpose(1, 2, 0)
            depth_gt_np = depth_gt.squeeze().cpu().numpy()

            # Image list and titles
            images = [image_numpy, depth_gt_np]
            titles = ["Original Image", "Ground Truth"]
            flag = False

            for _model_name in model_names:
                predict_depth_path = os.path.join(res_root, _model_name, dataset_name, f"{id}.png")
                predict_depth = cv2.imread(predict_depth_path, cv2.IMREAD_GRAYSCALE)
                if predict_depth is None:
                    print(f"File not found: {predict_depth_path}")
                    flag = True
                    break
                predict_depth = predict_depth.astype(np.float32) / 255.0
                images.append(predict_depth)
                titles.append(_model_name)
            if flag:
                continue

            # Create a grid for two rows and three columns
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            for i, ax in enumerate(axes.flatten()):
                if i < len(images):
                    # Resize the image to 768x768
                    resized_image = np.array(Image.fromarray(images[i]).resize((768, 768)))
                    ax.imshow(resized_image, cmap='viridis' if i > 0 else None)
                    ax.set_title(titles[i])
                ax.axis("off")  # Turn off axes for all subplots

            # Adjust layout and save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(save_root, f"{id}.png"))
            plt.close(fig)
