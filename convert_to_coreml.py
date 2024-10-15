import logging
import math
import numpy as np

import coremltools as ct
from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import upsample_bilinear2d
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op

import torch
from torch import nn
from torch.nn import functional as F

from matplotlib import pyplot as plt
from typing import Dict, Tuple

from src.depth_pro.depth_pro import (
    create_model_and_transforms,
    create_backbone_model,
    DepthProConfig
)
from src.depth_pro.network.decoder import MultiresConvDecoder
from src.depth_pro.network.encoder import DepthProEncoder
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

CONFIG_DICT: Dict[str, DepthProConfig] = {
    "large_192": DepthProConfig(
        patch_encoder_preset="dinov2l16_192",
        image_encoder_preset="dinov2l16_192",
        checkpoint_uri="./checkpoints/depth_pro.pt",
        decoder_features=256,
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_192",
        encoder_scale_size=(192, 192),
        head_paddings=[1, 0, 1, 0],
        fov_head_paddings=[1, 2, 3, 0],
    ),
    "large_288": DepthProConfig(
        patch_encoder_preset="dinov2l16_288",
        image_encoder_preset="dinov2l16_288",
        checkpoint_uri="./checkpoints/depth_pro.pt",
        decoder_features=256,
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_288",
        encoder_scale_size=(288, 288),
        head_paddings=[1, 0, 1, 0],
        fov_head_paddings=[1, 1, 2, 0],
    ),
    "large_384": DepthProConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        checkpoint_uri="./checkpoints/depth_pro.pt",
        decoder_features=256,
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_384",
        encoder_scale_size=(384, 384),
        head_paddings=[1, 0, 1, 0],
        fov_head_paddings=[1, 1, 1, 0],
    ),
}

class DepthDecoder(nn.Module):
    def __init__(self, head: nn.Module, fov: FOVNetwork, encoder_scale_size: (int, int)):
        super(DepthDecoder, self).__init__()
        self.head = head
        self.fov = fov
        self.encoder_scale_size = encoder_scale_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs[0]
        features = inputs[1]
        features_0 = inputs[2]

        # execute fov.forward locally with a different scale_factor
        # fov_deg = self.fov.forward(x, features_0.detach())
        if hasattr(self.fov, "encoder"):
            x = F.interpolate(
                x,
                size=self.encoder_scale_size,
                #scale_factor=self.encoder_scale_factor,
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

class DepthProScaled(nn.Module):
    def __init__(self, transform: nn.Module, encoder: DepthProEncoder, decoder: MultiresConvDecoder, depth: DepthDecoder):
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

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.size = size
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.size, mode=self.mode, align_corners=False)
        return x

def create_scaled_model(config: DepthProConfig) -> DepthProScaled:
    patch_encoder, patch_encoder_config = create_backbone_model(preset = config.patch_encoder_preset)
    image_encoder, _ = create_backbone_model(preset = config.image_encoder_preset)
    fov_encoder, _ = create_backbone_model(preset = config.fov_encoder_preset)
    # fov_encoder = None

    dims_encoder = patch_encoder_config.encoder_feature_dims
    hook_block_ids = patch_encoder_config.encoder_feature_layer_ids
    encoder = DepthProEncoder(
        dims_encoder=dims_encoder,
        patch_encoder=patch_encoder,
        image_encoder=image_encoder,
        hook_block_ids=hook_block_ids,
        decoder_features=config.decoder_features,
    )

    decoder = MultiresConvDecoder(
        dims_encoder=[config.decoder_features] + list(encoder.dims_encoder),
        dim_decoder=config.decoder_features,
    )

    num_features = config.decoder_features
    fov = FOVNetwork(num_features=num_features, fov_encoder=fov_encoder)
    # Create FOV head.
    fov_head0 = [
        nn.Conv2d(
            num_features, num_features // 2, kernel_size=3, stride=2, padding=config.fov_head_paddings[0]
        ),  # 128 x 24 x 24
        nn.ReLU(True),
    ]
    fov_head = [
        nn.Conv2d(
            num_features // 2, num_features // 4, kernel_size=3, stride=2, padding=config.fov_head_paddings[1]
        ),  # 64 x 12 x 12
        nn.ReLU(True),
        nn.Conv2d(
            num_features // 4, num_features // 8, kernel_size=3, stride=2, padding=config.fov_head_paddings[2]
        ),  # 32 x 6 x 6
        nn.ReLU(True),
        nn.Conv2d(num_features // 8, 1, kernel_size=6, stride=1, padding=config.fov_head_paddings[3]),
    ]
    if fov_encoder is not None:
        fov.encoder = nn.Sequential(
            fov_encoder, nn.Linear(fov_encoder.embed_dim, num_features // 2)
        )
        fov.downsample = nn.Sequential(*fov_head0)
    else:
        fov_head = fov_head0 + fov_head
    fov.head = nn.Sequential(*fov_head)
    # fov = None

    last_dims = (32, 1)
    dim_decoder = config.decoder_features
    head = nn.Sequential(
        nn.Conv2d(
            dim_decoder, dim_decoder // 2, kernel_size=3, stride=1, padding=config.head_paddings[0]
        ),
        nn.ConvTranspose2d(
            in_channels=dim_decoder // 2,
            out_channels=dim_decoder // 2,
            kernel_size=2,
            stride=2,
            padding=config.head_paddings[1],
            bias=True,
        ),
        nn.Conv2d(
            dim_decoder // 2,
            last_dims[0],
            kernel_size=3,
            stride=1,
            padding=config.head_paddings[2],
        ),
        nn.ReLU(True),
        nn.Conv2d(last_dims[0], last_dims[1], kernel_size=1, stride=1, padding=config.head_paddings[3]),
        nn.ReLU(),
    )

    # Set the final convolution layer's bias to be 0.
    head[4].bias.data.fill_(0)

    # from depth_pro.py
    transform = nn.Sequential(
        #[
            #ToTensor(),
            #Lambda(lambda x: x.to(device)),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            Interpolate(
                size=(encoder.img_size, encoder.img_size),
                mode="bilinear"
            ),
            ConvertImageDtype(torch.float32),
        #]
    )

    depth = DepthDecoder(head, fov, config.encoder_scale_size)
    load_state_dict(depth, config)

    model = DepthProScaled(transform, encoder, decoder, depth)
    load_state_dict(model, config)

    return model

def load_state_dict(model: nn.Module, config: DepthProConfig):
    checkpoint_uri = config.checkpoint_uri
    state_dict = torch.load(checkpoint_uri, map_location="cpu")
    _, _ = model.load_state_dict(
        state_dict=state_dict, strict=False
    )

def load_and_show_examples(models: tuple[DepthProScaled]):
    plt.ion()
    fig = plt.figure()
    ax_rgb = fig.add_subplot(1, 1 + len(models), 1)

    image, _, _ = load_rgb("data/example.jpg")
    ax_rgb.imshow(image)

    for index in range(len(models)):
        model_run = Compose([ToTensor(), Lambda(lambda x: x.to(torch.device("cpu"))), models[index]])
        depth_map = model_run(image).detach().cpu().numpy().squeeze()

        ax_disp = fig.add_subplot(1, 1 + len(models), 2 + index)
        ax_disp.imshow(depth_map, cmap="turbo")

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show(block=True)

def save_coreml_packages(model: DepthProScaled):
    transform = nn.Sequential(
        #[
            #ToTensor(),
            #Lambda(lambda x: x.to(device)),
            Normalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5]),
            Interpolate(
                size=(model.encoder.img_size, model.encoder.img_size),
                mode="bilinear"
            ),
            ConvertImageDtype(torch.float16),
        #]
    )
    save_mlpackage(transform, [[1, 3, 1080, 1920]], "DepthPro_transform", True)
    save_mlpackage(model.encoder, [[1, 3, 768, 768]], "DepthPro_encoder")
    save_mlpackage(model.decoder, [[1, 256, 288, 288], [1, 256, 144, 144], [1, 512, 72, 72], [1, 1024, 24, 24], [1, 1024, 24, 24]], "DepthPro_decoder")
    save_mlpackage(model.depth, [[1, 3, 768, 768], [1, 256, 288, 288], [1, 256, 24, 24]], "DepthPro_depth")
    save_mlpackage(model.depth.head, [[1, 256, 768, 768]], "DepthPro_head")

@register_torch_op()
def _upsample_bicubic2d_aa(context, node):
    upsample_bilinear2d(context, node)

# https://github.com/apple/coremltools/pull/2354 CoreMLTools 8.0 fix
from coremltools.converters.mil.frontend.torch.ops import _get_bindings, _get_inputs
from coremltools.converters.mil.frontend.torch.utils import TorchFrontend
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.ops.defs._utils import promote_input_dtypes
from coremltools.converters.mil.mil.var import Var
@register_torch_op(torch_alias=["concat"], override=True)
def cat(context, node):
    def is_tensor_empty(var: Var) -> bool:
        return np.any([size == 0 for size in var.shape])

    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, min_expected=1)
        nargs = len(inputs)

        xs = inputs[0]
        # PyTorch can have empty tensor, which is then ignored
        # However, CoreML does not allow such empty tensor, so remove them now
        if np.any([is_tensor_empty(x) for x in xs]):
            filtered_xs = [x for x in xs if not is_tensor_empty(x)]
            xs = filtered_xs if len(filtered_xs) > 0 else [xs[0]]

        dim = inputs[1] if nargs > 1 else 0

        return xs, dim

    def _parse_keyword_args(context, node, dim) -> Var:
        # Only torch.export may have kwargs
        if context.frontend != TorchFrontend.TORCHEXPORT:
            return dim

        dim = _get_kwinputs(context, node, "dim", default=[dim])[0]
        return dim

    xs, dim = _parse_positional_args(context, node)
    dim = _parse_keyword_args(context, node, dim)

    concat = mb.concat(values=promote_input_dtypes(xs), axis=dim, name=node.name)
    context.add(concat)

def save_mlpackage(G, shapes, name, image_type = False):
    G.eval()
    G_inputs = []
    convert_inputs = []
    for shape in shapes:
        G_inputs.append(torch.randn(shape))
        convert_inputs.append(ct.TensorType(shape=shape, dtype=np.float16) if image_type == False else ct.ImageType(shape=shape, color_layout=ct.colorlayout.RGB))
    G_trace = torch.jit.trace(G, G_inputs if len(G_inputs) == 1 else [G_inputs])
    G_model = ct.convert(
        G_trace,
        inputs=convert_inputs if len(convert_inputs) <= 1 else [convert_inputs],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    G_model.save("out/" + name + ".mlpackage")

if __name__ == "__main__":
    model_192 = create_scaled_model(CONFIG_DICT["large_192"])
    model_288 = create_scaled_model(CONFIG_DICT["large_288"])
    model_384 = create_scaled_model(CONFIG_DICT["large_384"])
    load_and_show_examples((model_192, model_288, model_384))

    # save_coreml_packages(model_192)
