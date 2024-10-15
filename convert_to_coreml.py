import coremltools as ct
import logging
import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from matplotlib import pyplot as plt
from typing import Tuple

from src.depth_pro.depth_pro import (
    create_model_and_transforms,
    create_backbone_model,
    DEFAULT_MONODEPTH_CONFIG_DICT
)
from src.depth_pro.network.fov import FOVNetwork
from src.depth_pro.network.vit import resize_vit, resize_patch_embed
from src.depth_pro.utils import load_rgb

from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    Lambda,
    Normalize,
    ToTensor
)

"""
example.jpg fov_deg =
default 1536x1536: 48.4297
scaled 1024x1024: 49.8382
"""

class DepthProRun(nn.Module):
    def __init__(self, transform: nn.Module, encoder: nn.Module, decoder: nn.Module, depth: nn.Module):
        super().__init__()
        self.transform = transform
        self.encoder = encoder
        self.decoder = decoder
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 3:
            x = x.unsqueeze(0)
        image = self.transform(x)
        encodings = self.encoder(image)
        features, features_0 = self.decoder(encodings)
        depth = self.depth([image, features, features_0])
        return depth

class Depth(nn.Module):
    def __init__(self, head: nn.Module, fov: nn.Module):
        super(Depth, self).__init__()
        self.head = head
        self.fov = fov

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs[0]
        features = inputs[1]
        features_0 = inputs[2]

        _, _, H, W = features_0.shape
        if H != 48 or W != 48:
            features_0 = F.interpolate(
                features_0,
                size=(48, 48),
                mode="bilinear",
                align_corners=False,
            )

        # execute fov.forward locally with a different scale_factor
        # fov_deg = self.fov.forward(x, features_0.detach())
        if hasattr(self.fov, "encoder"):
            x = F.interpolate(
                x,
                size=None,
                # result size needs to be 374
                scale_factor=0.375,
                mode="bilinear",
                align_corners=False,
            )
            x = self.fov.encoder(x)[:, 1:].permute(0, 2, 1)
            lowres_feature = self.fov.downsample(features_0.detach())
            x = x.reshape_as(lowres_feature) + lowres_feature
        else:
            x = features_0.detach()
        fov_deg = self.fov.head(x)
        f_px = 0.5 * torch.tan(math.pi * fov_deg.to(torch.float) / 360.0)

        canonical_inverse_depth = self.head(features)
        inverse_depth = canonical_inverse_depth * f_px
        depth = 1.0 / inverse_depth.clamp(min=1e-4, max=1e4)
        return depth

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.size = size
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.size, mode=self.mode, align_corners=False)
        return x

def save_mlpackage(G, shapes, name):
    G.eval()
    G_inputs = []
    convert_inputs = []
    for shape in shapes:
        G_inputs.append(torch.randn(shape))
        convert_inputs.append(ct.TensorType(shape=shape, dtype=np.float16))
    G_trace = torch.jit.trace(G, G_inputs if len(G_inputs) == 1 else [G_inputs])
    G_model = ct.convert(
        G_trace,
        inputs=convert_inputs if len(convert_inputs) <= 1 else [convert_inputs],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    G_model.save("out/" + name + ".mlpackage")

def create_scaled_model() -> Tuple[nn.Module, nn.Module, nn.Module]:
    # from run.py
    model, _ = create_model_and_transforms(
        device=torch.device("cpu"),
        precision=torch.float32,
    )

    # resize to 256x4 = 1024x1024 input image
    new_img_size = (256, 256)

    model.encoder.patch_encoder = resize_patch_embed(model.encoder.patch_encoder)
    model.encoder.patch_encoder = resize_vit(model.encoder.patch_encoder, img_size=new_img_size)
    model.encoder.image_encoder = resize_patch_embed(model.encoder.image_encoder)
    model.encoder.image_encoder = resize_vit(model.encoder.image_encoder, img_size=new_img_size)
    model.encoder.out_size = int(
        model.encoder.patch_encoder.patch_embed.img_size[0] // model.encoder.patch_encoder.patch_embed.patch_size[0]
    )

    # this is still under works to resize fov_encoder to 256x256 size too
    # fov_encoder, _ = create_backbone_model(preset = DEFAULT_MONODEPTH_CONFIG_DICT.fov_encoder_preset)
    # fov_encoder = resize_patch_embed(fov_encoder)
    # fov_encoder = resize_vit(fov_encoder, img_size=new_img_size)
    # model.fov = FOVNetwork(num_features=model.decoder.dim_decoder, fov_encoder=fov_encoder)

    # from depth_pro.py
    transform = nn.Sequential(
        #[
            #ToTensor(),
            #Lambda(lambda x: x.to(device)),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            Interpolate(
                size=(model.img_size, model.img_size),
                mode="bilinear"
            ),
            ConvertImageDtype(torch.float32),
        #]
    )

    depth = Depth(model.head, model.fov)
    return transform, model, depth

def load_and_show_example(transform: nn.Module, model: nn.Module, depth: nn.Module):
    image, _, _ = load_rgb("data/example.jpg")
    depth_pro_run = DepthProRun(transform, model.encoder, model.decoder, depth)

    depth_pro = Compose([ToTensor(), Lambda(lambda x: x.to(torch.device("cpu"))), depth_pro_run])
    depth_map = depth_pro(image).detach().cpu().numpy().squeeze()

    plt.ion()
    fig = plt.figure()
    ax_rgb = fig.add_subplot(121)
    ax_disp = fig.add_subplot(122)
    ax_rgb.imshow(image)
    ax_disp.imshow(depth_map, cmap="turbo")
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show(block=True)

def save_coreml_packages(transform: nn.Module, model: nn.Module, depth: nn.Module):
    save_mlpackage(transform, [[1, 3, 1024, 1024]], "DepthPro_transform")
    save_mlpackage(model.encoder, [[1, 3, 1024, 1024]], "DepthPro_encoder")
    save_mlpackage(model.decoder, [[1, 256, 512, 512], [1, 256, 256, 256], [1, 512, 128, 128], [1, 1024, 64, 64], [1, 1024, 32, 32]], "DepthPro_decoder")
    save_mlpackage(depth, [[1, 3, 1024, 1024], [1, 256, 512, 512], [1, 256, 32, 32]], "DepthPro_depth")

if __name__ == "__main__":
    transform, model, depth = create_scaled_model()
    load_and_show_example(transform, model, depth)
    save_coreml_packages(transform, model, depth)
