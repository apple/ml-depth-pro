import argparse
import os
import numpy as np
from PIL import Image
import open3d as o3d

import plotly.graph_objects as go


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize depth maps")
    parser.add_argument("--input_depth", type=str, required=True, help="Path to the input depth map (npz file)")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image (png file)")
    parser.add_argument("--focal_length", type=float, default=1500.0, help="Focal length of the camera")
    return parser.parse_args()


def visualize_depth_with_open3d(depth, image_rgb, focal_length=1500.0):
    # Convert numpy arrays to Open3D Image objects
    color_image = o3d.geometry.Image(image_rgb)
    depth_image = o3d.geometry.Image(depth)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image,
        depth_scale=1.0,  # Adjust this value based on your depth scale
        depth_trunc=100.0,  # Adjust this value based on your depth range
        convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            width=depth.shape[1],
            height=depth.shape[0],
            fx=focal_length,
            fy=focal_length,
            cx=depth.shape[1] / 2,
            cy=depth.shape[0] / 2
        )
    )

    # flip the orientation, so it looks upright, not upside-down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # Visualize the point cloud with coordinate frame
    # Set a smaller point size for the point cloud
    o3d.visualization.draw_geometries(
        [pcd, coordinate_frame], point_show_normal=True)  # Reduced point size to 1


if __name__ == "__main__":
    args = parse_args()

    # load depth map
    with np.load(args.input_depth) as data:
        depth = data["depth"]

    # load image
    img_pil = Image.open(args.input_image).convert("RGB")
    image = np.array(img_pil)

    visualize_depth_with_open3d(depth, image, focal_length=args.focal_length)
