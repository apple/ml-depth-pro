"""Copyright (C) 2024 Apple Inc. All Rights Reserved.

Dense Prediction Transformer Decoder architecture.

Implements a variant of Vision Transformers for Dense Prediction, https://arxiv.org/abs/2103.13413
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class MultiresConvDecoder(nn.Module):
    """Decoder for multi-resolution encodings."""

    def __init__(
        self,
        dims_encoder: Iterable[int],
        dim_decoder: int,
    ):
        """Initialize multiresolution convolutional decoder.

        Args:
        ----
            dims_encoder: Expected dims at each level from the encoder.
            dim_decoder: Dim of decoder features.

        """
        super().__init__()
        self.dims_encoder = list(dims_encoder)
        self.dim_decoder = dim_decoder
        self.dim_out = dim_decoder

        num_encoders = len(self.dims_encoder)

        # At the highest resolution, i.e. level 0, we apply projection w/ 1x1 convolution
        # when the dimensions mismatch. Otherwise we do not do anything, which is
        # the default behavior of monodepth.
        conv0 = (
            nn.Conv2d(self.dims_encoder[0], dim_decoder, kernel_size=1, bias=False)
            if self.dims_encoder[0] != dim_decoder
            else nn.Identity()
        )

        convs = [conv0]
        for i in range(1, num_encoders):
            convs.append(
                nn.Conv2d(
                    self.dims_encoder[i],
                    dim_decoder,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )

        self.convs = nn.ModuleList(convs)

        fusions = []
        for i in range(num_encoders):
            fusions.append(
                FeatureFusionBlock2d(
                    num_features=dim_decoder,
                    deconv=(i != 0),
                    batch_norm=False,
                )
            )
        self.fusions = nn.ModuleList(fusions)

    def forward(self, encodings: torch.Tensor) -> torch.Tensor:
        """Decode the multi-resolution encodings."""
        num_levels = len(encodings)
        num_encoders = len(self.dims_encoder)

        if num_levels != num_encoders:
            raise ValueError(
                f"Got encoder output levels={num_levels}, expected levels={num_encoders+1}."
            )

        # Project features of different encoder dims to the same decoder dim.
        # Fuse features from the lowest resolution (num_levels-1)
        # to the highest (0).
        features = self.convs[-1](encodings[-1])
        lowres_features = features
        features = self.fusions[-1](features)
        for i in range(num_levels - 2, -1, -1):
            features_i = self.convs[i](encodings[i])
            features = self.fusions[i](features, features_i)
        return features, lowres_features


class ResidualBlock(nn.Module):
    """Generic implementation of residual blocks.

    This implements a generic residual block from
        He et al. - Identity Mappings in Deep Residual Networks (2016),
        https://arxiv.org/abs/1603.05027
    which can be further customized via factory functions.
    """

    def __init__(self, residual: nn.Module, shortcut: nn.Module | None = None) -> None:
        """Initialize ResidualBlock."""
        super().__init__()
        self.residual = residual
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block."""
        delta_x = self.residual(x)

        if self.shortcut is not None:
            x = self.shortcut(x)

        return x + delta_x


class FeatureFusionBlock2d(nn.Module):
    """Feature fusion for DPT."""

    def __init__(
        self,
        num_features: int,
        deconv: bool = False,
        batch_norm: bool = False,
    ):
        """Initialize feature fusion block.

        Args:
        ----
            num_features: Input and output dimensions.
            deconv: Whether to use deconv before the final output conv.
            batch_norm: Whether to use batch normalization in resnet blocks.

        """
        super().__init__()

        self.resnet1 = self._residual_block(num_features, batch_norm)
        self.resnet2 = self._residual_block(num_features, batch_norm)

        self.use_deconv = deconv
        if deconv:
            self.deconv = nn.ConvTranspose2d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            )

        self.out_conv = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x0: torch.Tensor, x1: torch.Tensor | None = None) -> torch.Tensor:
        """Process and fuse input features."""
        x = x0

        if x1 is not None:
            res = self.resnet1(x1)
            x = self.skip_add.add(x, res)

        x = self.resnet2(x)

        if self.use_deconv:
            x = self.deconv(x)
        x = self.out_conv(x)

        return x

    @staticmethod
    def _residual_block(num_features: int, batch_norm: bool):
        """Create a residual block."""

        def _create_block(dim: int, batch_norm: bool) -> list[nn.Module]:
            layers = [
                nn.ReLU(False),
                nn.Conv2d(
                    num_features,
                    num_features,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not batch_norm,
                ),
            ]
            if batch_norm:
                layers.append(nn.BatchNorm2d(dim))
            return layers

        residual = nn.Sequential(
            *_create_block(dim=num_features, batch_norm=batch_norm),
            *_create_block(dim=num_features, batch_norm=batch_norm),
        )
        return ResidualBlock(residual)
