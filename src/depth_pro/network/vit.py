# Copyright (C) 2024 Apple Inc. All Rights Reserved.


try:
    from timm.layers import resample_abs_pos_embed
except ImportError as err:
    print("ImportError: {0}".format(err))
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def make_vit_b16_backbone(
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
            0 if getattr(model, "no_embed_class", False) else model.num_prefix_tokens
        ),
    )
    model.pos_embed = torch.nn.Parameter(pos_embed)

    return model


def resize_patch_embed(model: nn.Module, new_patch_size=(16, 16)) -> nn.Module:
    """Resample the ViT patch size to the given one."""
    # interpolate patch embedding
    if hasattr(model, "patch_embed"):
        old_patch_size = model.patch_embed.patch_size

        if (
            new_patch_size[0] != old_patch_size[0]
            or new_patch_size[1] != old_patch_size[1]
        ):
            patch_embed_proj = model.patch_embed.proj.weight
            patch_embed_proj_bias = model.patch_embed.proj.bias
            use_bias = True if patch_embed_proj_bias is not None else False
            _, _, h, w = patch_embed_proj.shape

            new_patch_embed_proj = torch.nn.functional.interpolate(
                patch_embed_proj,
                size=[new_patch_size[0], new_patch_size[1]],
                mode="bicubic",
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
                int(
                    model.patch_embed.img_size[0]
                    * new_patch_size[0]
                    / old_patch_size[0]
                ),
                int(
                    model.patch_embed.img_size[1]
                    * new_patch_size[1]
                    / old_patch_size[1]
                ),
            )

    return model
