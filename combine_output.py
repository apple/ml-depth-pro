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


def clip_array_percentile(array, lower_percentile=20, upper_percentile=80):
    """
    Clips the values in the NumPy array to the range defined by the lower and upper percentiles.

    Parameters:
        array (np.ndarray): Input array.
        lower_percentile (float): The lower percentile (default: 20).
        upper_percentile (float): The upper percentile (default: 80).

    Returns:
        np.ndarray: The clipped array.
    """
    # Calculate the percentile values
    lower_bound = np.percentile(array, lower_percentile)
    upper_bound = np.percentile(array, upper_percentile)

    # Clip the array
    clipped_array = np.clip(array, lower_bound, upper_bound)

    return clipped_array


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
            depth_gt_np = clip_array_percentile(depth_gt_np, 5, 95)
            depth_gt_np = depth_gt_np / 200

            # Image list and titles
            images = [image_numpy, depth_gt_np]
            titles = ["Original Image", "Ground Truth"]
            flag = False

            for _model_name in model_names:
                predict_depth_path = os.path.join(res_root, _model_name, dataset_name, f"{id}.png")
                predict_depth = cv2.imread(predict_depth_path, cv2.IMREAD_UNCHANGED)
                if len(predict_depth.shape) == 3:
                    predict_depth = cv2.cvtColor(predict_depth, cv2.COLOR_BGR2RGB)
                print(f"Shape of predict_depth: {predict_depth.shape}")
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
                    # Ensure the image is converted to uint8 if necessary
                    if images[i].dtype == np.float32:
                        image = (images[i] * 255).clip(0, 255).astype(np.uint8)
                    else:
                        image = images[i]
                    # Resize using OpenCV
                    resized_image = cv2.resize(image, (768, 768), interpolation=cv2.INTER_LINEAR)
                    ax.imshow(resized_image, cmap='viridis' if i not in [1, 4] else None)
                    ax.set_title(titles[i])
                ax.axis("off")  # Turn off axes for all subplots

            # Adjust layout and save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(save_root, f"{id}.png"))
            plt.close(fig)
