#!/usr/bin/env python3
"""Sample script to run DepthPro.

Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import argparse
import logging
from pathlib import Path
import time

import numpy as np
import PIL.Image
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from depth_pro import create_model_and_transforms, load_rgb

LOGGER = logging.getLogger(__name__)

def get_torch_device() -> torch.device:
    """Get the Torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def run(args):
    """Run Depth Pro on a sample image or directory of images."""
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    device = get_torch_device()
    LOGGER.info(f"Using device: {device}")

    # Load model
    start_time = time.time()
    model, transform = create_model_and_transforms(device=device, precision=torch.half)
    model.eval()
    LOGGER.info(f"Model loaded in {time.time() - start_time:.2f} seconds")

    image_paths = list(args.image_path.glob("**/*")) if args.image_path.is_dir() else [args.image_path]
    relative_path = args.image_path if args.image_path.is_dir() else args.image_path.parent

    if not args.skip_display:
        plt.ion()
        fig, (ax_rgb, ax_disp) = plt.subplots(1, 2, figsize=(12, 6))

    total_inference_time = 0
    for image_path in tqdm(image_paths, desc="Processing images"):
        if not image_path.is_file() or image_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
            continue

        try:
            LOGGER.info(f"Loading image {image_path} ...")
            image, _, f_px = load_rgb(image_path)
        except Exception as e:
            LOGGER.error(f"Error loading {image_path}: {str(e)}")
            continue

        # Run prediction
        start_time = time.time()
        with torch.no_grad():
            prediction = model.infer(transform(image), f_px=f_px)
        inference_time = time.time() - start_time
        total_inference_time += inference_time

        depth = prediction["depth"].cpu().numpy().squeeze()
        if f_px is not None:
            LOGGER.info(f"Focal length (from exif): {f_px:0.2f}")
        elif prediction["focallength_px"] is not None:
            focallength_px = prediction["focallength_px"].cpu().item()
            LOGGER.info(f"Estimated focal length: {focallength_px:0.2f}")

        inverse_depth = 1 / np.clip(depth, 0.1, 250)
        inverse_depth_normalized = (inverse_depth - inverse_depth.min()) / (inverse_depth.max() - inverse_depth.min())

        if args.output_path:
            output_file = args.output_path / image_path.relative_to(relative_path).with_suffix('')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save depth as NPZ
            np.savez_compressed(output_file, depth=depth)
            LOGGER.info(f"Depth map saved to: {output_file}.npz")

            # Save color-mapped depth as JPG
            cmap = plt.get_cmap('turbo')
            color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
            color_map_output_file = output_file.with_suffix('.jpg')
            PIL.Image.fromarray(color_depth).save(color_map_output_file, format="JPEG", quality=95)
            LOGGER.info(f"Color-mapped depth saved to: {color_map_output_file}")

        if not args.skip_display:
            ax_rgb.clear()
            ax_disp.clear()
            ax_rgb.imshow(image)
            ax_rgb.set_title("Input Image")
            ax_rgb.axis('off')
            im = ax_disp.imshow(inverse_depth_normalized, cmap="turbo")
            ax_disp.set_title(f"Depth Map (Inference time: {inference_time:.3f}s)")
            ax_disp.axis('off')
            plt.colorbar(im, ax=ax_disp, label="Inverse Depth", orientation="horizontal", pad=0.05)
            fig.tight_layout()
            plt.pause(0.1)

    avg_inference_time = total_inference_time / len(image_paths)
    LOGGER.info(f"Average inference time: {avg_inference_time:.3f} seconds per image")
    LOGGER.info("Depth estimation completed!")

    if not args.skip_display:
        plt.ioff()
        plt.show()

def main():
    """Run DepthPro inference example."""
    parser = argparse.ArgumentParser(description="Inference script for DepthPro with PyTorch models.")
    parser.add_argument("-i", "--image-path", type=Path, required=True, help="Path to input image or directory.")
    parser.add_argument("-o", "--output-path", type=Path, help="Path to store output files.")
    parser.add_argument("--skip-display", action="store_true", help="Skip matplotlib display.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose output.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference (default: 1)")
    
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
