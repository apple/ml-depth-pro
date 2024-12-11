import json
import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import src.depth_pro.depth_pro as depth_pro
from dataset.HypersimDataset import HypersimDataset
from dataset.SintelDataset import SintelDataset
from dataset.utils import get_hdf5_array

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# dataset = HypersimDataset()
dataset = SintelDataset()
save_root = "./output/sintel"
os.makedirs(save_root, exist_ok=True)

for id, (image, depth_gt) in enumerate(dataset):
    image = image.unsqueeze(0)
    image_numpy = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    print(f"Image range: {np.min(image_numpy), np.max(image_numpy)}")

    # Run inference.
    prediction = model.infer((image - 0.5) * 2, f_px=None)
    print(f"prediction shape: {prediction['depth'].shape}")
    depth = prediction["depth"]  # Depth in [m].
    predict_depth_np = depth.cpu().numpy()
    # print(f"Img,prediction,gt shape: {image_numpy.shape},{predict_depth_np.shape},{depth_gt.shape}")
    focallength_px = prediction["focallength_px"]  # Focal length in pixels.

    # Normalize the depth maps for visualization
    predict_depth_vis = predict_depth_np
    print(depth_gt.shape)
    depth_gt = depth_gt.squeeze().cpu().numpy()
    depth_gt_vis = depth_gt

    # Create the figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # Plot input image
    axes[0].imshow(image_numpy)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Plot predicted depth
    axes[1].imshow(predict_depth_vis.squeeze(), cmap="viridis")
    axes[1].set_title("Predicted Depth")
    axes[1].axis("off")

    # Plot ground truth depth
    axes[2].imshow(depth_gt_vis.squeeze(), cmap="viridis")
    axes[2].set_title("Ground Truth Depth")
    axes[2].axis("off")

    # Save and show the figure
    output_path = os.path.join(save_root, f"{id}.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Visualization saved to {output_path}")
