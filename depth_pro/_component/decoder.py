from copy import deepcopy
from typing import Self

import torch
from torch import nn


class Decoder(nn.Module):
    """Decoder for multi-resolution encodings."""

    dims_encoder: list[int]
    dim_decoder: int
    dim_out: int

    convs: nn.ModuleList
    fusions: nn.ModuleList

    def __init__(
        self,
        dims_encoder: list[int],
        dim_decoder: int,
    ):
        """Initialize multiresolution convolutional decoder.

        Args:
        ----
            dims_encoder: Expected dims at each level from the encoder.
            dim_decoder: Dim of decoder features.

        """
        super().__init__()

        self.dims_encoder = dims_encoder
        self.dim_decoder = dim_decoder
        self.dim_out = dim_decoder

        n_encoders = len(dims_encoder)

        # At the highest resolution, i.e. level 0, we apply projection w/ 1x1 convolution
        # when the dimensions mismatch. Otherwise we do not do anything, which is
        # the default behavior of monodepth.
        conv0 = (
            nn.Conv2d(dims_encoder[0], dim_decoder, kernel_size=1, bias=False)
            if self.dims_encoder[0] != dim_decoder
            else nn.Identity()
        )

        convs = [conv0] + [
            nn.Conv2d(
                in_channels,
                dim_decoder,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            for in_channels in dims_encoder[1:]
        ]
        self.convs = nn.ModuleList(convs)

        fusions = [
            FeatureFusionBlock2d(
                features=dim_decoder,
                use_deconv=False,
                batch_norm=False,
            )
        ] + [
            FeatureFusionBlock2d(
                features=dim_decoder,
                use_deconv=True,
                batch_norm=False,
            )
            for i in range(1, n_encoders)
        ]
        self.fusions = nn.ModuleList(fusions)

    def forward(self, encodings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode the multi-resolution encodings."""
        num_levels = len(encodings)
        num_encoders = len(self.dims_encoder)

        if num_levels != num_encoders:
            raise ValueError(
                f'Got encoder output levels={num_levels}, expected levels={num_encoders+1}.'
            )

        # Project features of different encoder dims to the same decoder dim.
        # Fuse features from the lowest resolution (num_levels-1)
        # to the highest (0).
        features = self.convs[-1](encodings[-1])
        low_resolution_features = features
        features = self.fusions[-1](features)

        for i in range(num_levels - 2, -1, -1):
            features_i = self.convs[i](encodings[i])
            features = self.fusions[i](features, features_i)

        return features, low_resolution_features


class ResidualBlock(nn.Module):
    """Generic implementation of residual blocks.

    This implements a generic residual block from
        He et al. - Identity Mappings in Deep Residual Networks (2016),
        https://arxiv.org/abs/1603.05027
    which can be further customized via factory functions.
    """

    residual: nn.Module
    shortcut: nn.Module | None

    def __init__(self, residual: nn.Module, shortcut: nn.Module | None = None) -> None:
        """Initialize ResidualBlock."""
        super().__init__()
        self.residual = residual
        self.shortcut = shortcut

    @classmethod
    def with_shape(cls, n: int, batch_norm: bool) -> Self:
        layers = [
            nn.ReLU(inplace=False),
            nn.Conv2d(
                n,
                n,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not batch_norm,
            ),
        ]

        if batch_norm:
            layers.append(nn.BatchNorm2d(n))

        return cls(
            nn.Sequential(
                *deepcopy(layers),
                *layers,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block."""
        delta_x = self.residual(x)

        if self.shortcut is not None:
            x = self.shortcut(x)

        return x + delta_x


class FeatureFusionBlock2d(nn.Module):
    """Feature fusion for DPT."""

    features: int
    use_deconv: bool

    skip_add: torch.ao.nn.quantized.FloatFunctional

    resnet1: ResidualBlock
    resnet2: ResidualBlock

    deconv: nn.ConvTranspose2d
    out_conv: nn.Conv2d

    def __init__(
        self,
        features: int,
        use_deconv: bool = False,
        batch_norm: bool = False,
    ):
        """Initialize feature fusion block.

        Args:
        ----
            features: Input and output dimensions.
            deconv: Whether to use deconv before the final output conv.
            batch_norm: Whether to use batch normalization in resnet blocks.

        """
        super().__init__()

        self.features = features
        self.use_deconv = use_deconv
        self.skip_add = torch.ao.nn.quantized.FloatFunctional()

        self.resnet1 = ResidualBlock.with_shape(features, batch_norm)
        self.resnet2 = ResidualBlock.with_shape(features, batch_norm)

        if use_deconv:
            self.deconv = nn.ConvTranspose2d(
                in_channels=features,
                out_channels=features,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            )

        self.out_conv = nn.Conv2d(
            features,
            features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

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
