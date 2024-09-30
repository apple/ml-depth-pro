# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# Factory functions to build and load ViT models.


from __future__ import annotations

import logging
import types
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import timm
import torch
import torch.nn as nn

from .vit import (
    forward_features_eva_fixed,
    make_vit_b16_backbone,
    resize_patch_embed,
    resize_vit,
)

LOGGER = logging.getLogger(__name__)


ViTPreset = Literal[
    "dinov2l16_384",
]


@dataclass
class ViTConfig:
    """Configuration for ViT."""

    in_chans: int
    embed_dim: int

    img_size: int = 384
    patch_size: int = 16

    # In case we need to rescale the backbone when loading from timm.
    timm_preset: Optional[str] = None
    timm_img_size: int = 384
    timm_patch_size: int = 16

    # The following 2 parameters are only used by DPT.  See dpt_factory.py.
    encoder_feature_layer_ids: List[int] = None
    """The layers in the Beit/ViT used to constructs encoder features for DPT."""
    encoder_feature_dims: List[int] = None
    """The dimension of features of encoder layers from Beit/ViT features for DPT."""


VIT_CONFIG_DICT: Dict[ViTPreset, ViTConfig] = {
    "dinov2l16_384": ViTConfig(
        in_chans=3,
        embed_dim=1024,
        encoder_feature_layer_ids=[5, 11, 17, 23],
        encoder_feature_dims=[256, 512, 1024, 1024],
        img_size=384,
        patch_size=16,
        timm_preset="vit_large_patch14_dinov2",
        timm_img_size=518,
        timm_patch_size=14,
    ),
}


def create_vit(
    preset: ViTPreset,
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
    config = VIT_CONFIG_DICT[preset]

    img_size = (config.img_size, config.img_size)
    patch_size = (config.patch_size, config.patch_size)

    if "eva02" in preset:
        model = timm.create_model(config.timm_preset, pretrained=use_pretrained)
        model.forward_features = types.MethodType(forward_features_eva_fixed, model)
    else:
        model = timm.create_model(
            config.timm_preset, pretrained=use_pretrained, dynamic_img_size=True
        )
    model = make_vit_b16_backbone(
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
        state_dict = torch.load(checkpoint_uri, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict=state_dict, strict=False
        )

        if len(unexpected_keys) != 0:
            raise KeyError(f"Found unexpected keys when loading vit: {unexpected_keys}")
        if len(missing_keys) != 0:
            raise KeyError(f"Keys are missing when loading vit: {missing_keys}")

    LOGGER.info(model)
    return model.model
