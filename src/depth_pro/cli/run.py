#  Add Configurable Options for Model Precision and Device
# Description: Allow users to specify precision (e.g., float32, float16) and device (CPU, CUDA, MPS) as arguments for more flexibility.
# Code Update: Add arguments in argparse and modify get_torch_device and create_model_and_transforms().
#  parser.add_argument
(
     "--precision",
     type=str,
     choices=["float32", "float16"],
     default="float16",
     help="Precision for model inference, choose between 'float32' or 'float16'."
 )
 parser.add_argument(
     "--device",
     type=str,
     choices=["cpu", "cuda", "mps"],
     default="cuda" if torch.cuda.is_available() else "cpu",
     help="Device to run the model on. Default is 'cuda' if available."
 )
# 2. Support Batch Processing of Images
# Description: Extend run() to accept a directory of images for batch processing.
# Code Update: Update image_paths logic to handle directories and provide support for tqdm progress tracking.
 if args.image_path.is_dir():
     image_paths = list(args.image_path.glob("*.jpg")) + list(args.image_path.glob("*.png"))
 ```

### 3. **Add Error Handling and Warnings**
- **Description**: Improve robustness by adding error handling and warnings for potential issues.
- **Code Update**: Add `try-except` blocks around file I/O and GPU checks.

```python
 try:
     depth = prediction["depth"].detach().cpu().numpy().squeeze()
 except AttributeError as e:
     LOGGER.warning(f"Error processing depth: {e}")
     continue

 parser.add_argument(
     "--output-format",
     type=str,
     choices=["npz", "png", "tiff"],
     default="npz",
     help="Output format for saving depth maps."
 )
 parser.add_argument(
     "--colormap",
     type=str,
     default="turbo",
     help="Colormap for depth visualization. E.g., 'viridis', 'plasma', 'turbo'."
 )
 parser.add_argument(
     "--log-file",
     type=Path,
     help="Path to save log file."
 )
