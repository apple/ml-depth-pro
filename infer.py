import json
import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import src.depth_pro.depth_pro as depth_pro
from dataset.HypersimDataset import HypersimDataset
from dataset.utils import get_hdf5_array

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

dataset = HypersimDataset()
meta_json = dataset.meta_json
image_paths = []
depth_paths = []
with open(meta_json, "r", encoding="utf-8") as infile:
    for line in infile:
        entry = json.loads(line)
        image_paths.append(entry["img_path"])
        depth_paths.append(entry["depth_path"])

print(f"Total images: {len(image_paths)}, Total depths: {len(depth_paths)}")

os.makedirs("./output", exist_ok=True)

for id in range(len(image_paths)):
    # Load and preprocess an image.
    image_path = image_paths[id]
    depth_path = depth_paths[id]
    image = get_hdf5_array(image_path)
    depth_gt = get_hdf5_array(depth_path)
    image = np.clip(image, 0.0, 1.0)
    depth_gt = np.clip(depth_gt, 0.0, 200.0)
    f_px = None
    image = transform(image)
    print(f"Image shape: {image.shape}")

    image = image.unsqueeze(0)
    image_numpy = image.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5

    print(f"Image range: {np.min(image_numpy), np.max(image_numpy)}")

    # Run inference.
    prediction = model.infer(image, f_px=f_px)
    print(f"prediction shape: {prediction['depth'].shape}")
    depth = prediction["depth"]  # Depth in [m].
    predict_depth_np = depth.cpu().numpy()
    print(f"Img,prediction,gt shape: {image_numpy.shape},{predict_depth_np.shape},{depth_gt.shape}")
    focallength_px = prediction["focallength_px"]  # Focal length in pixels.

    # Normalize the depth maps for visualization
    predict_depth_vis = (predict_depth_np - np.min(predict_depth_np)) / (
            np.max(predict_depth_np) - np.min(predict_depth_np))
    depth_gt_vis = (depth_gt - np.min(depth_gt)) / (np.max(depth_gt) - np.min(depth_gt))

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
    output_path = os.path.join("./output", f"{id}.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Visualization saved to {output_path}")
