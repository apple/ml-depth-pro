import types
from dataclasses import dataclass
from typing import Literal

import timm
import torch
import torch.nn as nn
from timm.layers import resample_abs_pos_embed
from torch.utils.checkpoint import checkpoint

type Preset = Literal['dinov2l16_384',]


@dataclass
class Config:
    """Configuration for ViT."""

    in_chans: int
    embed_dim: int

    img_size: int = 384
    patch_size: int = 16

    # In case we need to rescale the backbone when loading from timm.
    timm_preset: str | None = None
    timm_img_size: int = 384
    timm_patch_size: int = 16

    # The following 2 parameters are only used by DPT.  See dpt_factory.py.
    encoder_feature_layer_ids: list[int] | None = None
    """The layers in the Beit/ViT used to constructs encoder features for DPT."""
    encoder_feature_dims: list[int] | None = None
    """The dimension of features of encoder layers from Beit/ViT features for DPT."""


PRESETS: dict[Preset, Config] = {
    'dinov2l16_384': Config(
        in_chans=3,
        embed_dim=1024,
        encoder_feature_layer_ids=[5, 11, 17, 23],
        encoder_feature_dims=[256, 512, 1024, 1024],
        img_size=384,
        patch_size=16,
        timm_preset='vit_large_patch14_dinov2',
        timm_img_size=518,
        timm_patch_size=14,
    ),
}


def create(
    preset: Preset,
    use_pretrained: bool = False,
    checkpoint_uri: str | None = None,
    use_grad_checkpointing: bool = False,
) -> nn.Module:
    """Create and load a VIT backbone module.

    Args:
    ----
        preset: The VIT preset to load the pre-defined config.
        use_pretrained: Load pretrained weights if True, default is False.
        checkpoint_uri: Checkpoint to load the wights from.
        use_grad_checkpointing: Use grandient checkpointing.

    Returns:
    -------
        A Torch ViT backbone module.

    """
    config = PRESETS[preset]

    img_size = (config.img_size, config.img_size)
    patch_size = (config.patch_size, config.patch_size)

    if 'eva02' in preset:
        model = timm.create_model(config.timm_preset, pretrained=use_pretrained)
        model.forward_features = types.MethodType(forward_features_eva_fixed, model)
    else:
        model = timm.create_model(
            config.timm_preset, pretrained=use_pretrained, dynamic_img_size=True
        )
    model = vit_b16_backbone(
        model,
        encoder_feature_dims=config.encoder_feature_dims,
        encoder_feature_layer_ids=config.encoder_feature_layer_ids,
        vit_features=config.embed_dim,
        use_grad_checkpointing=use_grad_checkpointing,
    )
    if config.patch_size != config.timm_patch_size:
        model.model = resize_patch_embed(model.model, new_patch_size=patch_size)
    if config.img_size != config.timm_img_size:
        model.model = resize_vit(model.model, img_size=img_size)

    if checkpoint_uri is not None:
        state_dict = torch.load(checkpoint_uri, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict=state_dict, strict=False
        )

        if len(unexpected_keys) != 0:
            raise KeyError(f'Found unexpected keys when loading vit: {unexpected_keys}')
        if len(missing_keys) != 0:
            raise KeyError(f'Keys are missing when loading vit: {missing_keys}')

    return model.model


def vit_b16_backbone(
    model,
    encoder_feature_dims,
    encoder_feature_layer_ids,
    vit_features,
    start_index=1,
    use_grad_checkpointing=False,
) -> nn.Module:
    """Make a ViTb16 backbone for the DPT model."""
    if use_grad_checkpointing:
        model.set_grad_checkpointing()

    vit_model = nn.Module()
    vit_model.hooks = encoder_feature_layer_ids
    vit_model.model = model
    vit_model.features = encoder_feature_dims
    vit_model.vit_features = vit_features
    vit_model.model.start_index = start_index
    vit_model.model.patch_size = vit_model.model.patch_embed.patch_size
    vit_model.model.is_vit = True
    vit_model.model.forward = vit_model.model.forward_features

    return vit_model


def forward_features_eva_fixed(self, x):
    """Encode features."""
    x = self.patch_embed(x)
    x, rot_pos_embed = self._pos_embed(x)

    for blk in self.blocks:
        if self.grad_checkpointing:
            x = checkpoint(blk, x, rot_pos_embed)
        else:
            x = blk(x, rot_pos_embed)

    x = self.norm(x)

    return x


def resize_vit(model: nn.Module, img_size) -> nn.Module:
    """Resample the ViT module to the given size."""
    patch_size = model.patch_embed.patch_size
    model.patch_embed.img_size = img_size
    grid_size = tuple([s // p for s, p in zip(img_size, patch_size)])
    model.patch_embed.grid_size = grid_size

    pos_embed = resample_abs_pos_embed(
        model.pos_embed,
        grid_size,  # img_size
        num_prefix_tokens=(
            0 if getattr(model, 'no_embed_class', False) else model.num_prefix_tokens
        ),
    )
    model.pos_embed = torch.nn.Parameter(pos_embed)

    return model


def resize_patch_embed(model: nn.Module, new_patch_size=(16, 16)) -> nn.Module:
    """Resample the ViT patch size to the given one."""
    # interpolate patch embedding

    if not hasattr(model, 'patch_embed'):
        return model

    old_patch_size = model.patch_embed.patch_size

    if new_patch_size[0] == old_patch_size[0] and new_patch_size[1] == old_patch_size[1]:
        return model

    patch_embed_proj = model.patch_embed.proj.weight
    patch_embed_proj_bias = model.patch_embed.proj.bias
    use_bias = True if patch_embed_proj_bias is not None else False
    _, _, h, w = patch_embed_proj.shape

    new_patch_embed_proj = torch.nn.functional.interpolate(
        patch_embed_proj,
        size=[new_patch_size[0], new_patch_size[1]],
        mode='bicubic',
        align_corners=False,
    )
    new_patch_embed_proj = (
        new_patch_embed_proj * (h / new_patch_size[0]) * (w / new_patch_size[1])
    )

    model.patch_embed.proj = nn.Conv2d(
        in_channels=model.patch_embed.proj.in_channels,
        out_channels=model.patch_embed.proj.out_channels,
        kernel_size=new_patch_size,
        stride=new_patch_size,
        bias=use_bias,
    )

    if use_bias:
        model.patch_embed.proj.bias = patch_embed_proj_bias

    model.patch_embed.proj.weight = torch.nn.Parameter(new_patch_embed_proj)

    model.patch_size = new_patch_size
    model.patch_embed.patch_size = new_patch_size
    model.patch_embed.img_size = (
        int(model.patch_embed.img_size[0] * new_patch_size[0] / old_patch_size[0]),
        int(model.patch_embed.img_size[1] * new_patch_size[1] / old_patch_size[1]),
    )

    return model
