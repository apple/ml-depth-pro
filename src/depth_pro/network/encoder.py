# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# DepthProEncoder combining patch and image encoders.

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthProEncoder(nn.Module):
    """DepthPro Encoder.

    An encoder aimed at creating multi-resolution encodings from Vision Transformers.
    """

    def __init__(
        self,
        dims_encoder: Iterable[int],
        patch_encoder: nn.Module,
        image_encoder: nn.Module,
        hook_block_ids: Iterable[int],
        decoder_features: int,
    ):
        """Initialize DepthProEncoder.

        The framework
            1. creates an image pyramid,
            2. generates overlapping patches with a sliding window at each pyramid level,
            3. creates batched encodings via vision transformer backbones,
            4. produces multi-resolution encodings.

        Args:
        ----
            img_size: Backbone image resolution.
            dims_encoder: Dimensions of the encoder at different layers.
            patch_encoder: Backbone used for patches.
            image_encoder: Backbone used for global image encoder.
            hook_block_ids: Hooks to obtain intermediate features for the patch encoder model.
            decoder_features: Number of feature output in the decoder.

        """
        super().__init__()

        self.dims_encoder = list(dims_encoder)
        self.patch_encoder = patch_encoder
        self.image_encoder = image_encoder
        self.hook_block_ids = list(hook_block_ids)

        patch_encoder_embed_dim = patch_encoder.embed_dim
        image_encoder_embed_dim = image_encoder.embed_dim

        self.out_size = int(
            patch_encoder.patch_embed.img_size[0] // patch_encoder.patch_embed.patch_size[0]
        )

        def _create_project_upsample_block(
            dim_in: int,
            dim_out: int,
            upsample_layers: int,
            dim_int: Optional[int] = None,
        ) -> nn.Module:
            if dim_int is None:
                dim_int = dim_out
            # Projection.
            blocks = [
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=dim_int,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            ]

            # Upsampling.
            blocks += [
                nn.ConvTranspose2d(
                    in_channels=dim_int if i == 0 else dim_out,
                    out_channels=dim_out,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=False,
                )
                for i in range(upsample_layers)
            ]

            return nn.Sequential(*blocks)

        self.upsample_latent0 = _create_project_upsample_block(
            dim_in=patch_encoder_embed_dim,
            dim_int=self.dims_encoder[0],
            dim_out=decoder_features,
            upsample_layers=3,
        )
        self.upsample_latent1 = _create_project_upsample_block(
            dim_in=patch_encoder_embed_dim, dim_out=self.dims_encoder[0], upsample_layers=2
        )

        self.upsample0 = _create_project_upsample_block(
            dim_in=patch_encoder_embed_dim, dim_out=self.dims_encoder[1], upsample_layers=1
        )
        self.upsample1 = _create_project_upsample_block(
            dim_in=patch_encoder_embed_dim, dim_out=self.dims_encoder[2], upsample_layers=1
        )
        self.upsample2 = _create_project_upsample_block(
            dim_in=patch_encoder_embed_dim, dim_out=self.dims_encoder[3], upsample_layers=1
        )

        self.upsample_lowres = nn.ConvTranspose2d(
            in_channels=image_encoder_embed_dim,
            out_channels=self.dims_encoder[3],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )
        self.fuse_lowres = nn.Conv2d(
            in_channels=(self.dims_encoder[3] + self.dims_encoder[3]),
            out_channels=self.dims_encoder[3],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Obtain intermediate outputs of the blocks.
        self.patch_encoder.blocks[self.hook_block_ids[0]].register_forward_hook(
            self._hook0
        )
        self.patch_encoder.blocks[self.hook_block_ids[1]].register_forward_hook(
            self._hook1
        )

    def _hook0(self, model, input, output):
        self.backbone_highres_hook0 = output

    def _hook1(self, model, input, output):
        self.backbone_highres_hook1 = output

    @property
    def img_size(self) -> int:
        """Return the full image size of the SPN network."""
        return self.patch_encoder.patch_embed.img_size[0] * 4

    def _create_pyramid(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a 3-level image pyramid."""
        # Original resolution: 1536 by default.
        x0 = x

        # Middle resolution: 768 by default.
        x1 = F.interpolate(
            x, size=None, scale_factor=0.5, mode="bilinear", align_corners=False
        )

        # Low resolution: 384 by default, corresponding to the backbone resolution.
        x2 = F.interpolate(
            x, size=None, scale_factor=0.25, mode="bilinear", align_corners=False
        )

        return x0, x1, x2

    def split(self, x: torch.Tensor, overlap_ratio: float = 0.25) -> torch.Tensor:
        """Split the input into small patches with sliding window."""
        patch_size = 384
        patch_stride = int(patch_size * (1 - overlap_ratio))

        image_size = x.shape[-1]
        steps = int(math.ceil((image_size - patch_size) / patch_stride)) + 1

        x_patch_list = []
        for j in range(steps):
            j0 = j * patch_stride
            j1 = j0 + patch_size

            for i in range(steps):
                i0 = i * patch_stride
                i1 = i0 + patch_size
                x_patch_list.append(x[..., j0:j1, i0:i1])

        return torch.cat(x_patch_list, dim=0)

    def merge(self, x: torch.Tensor, batch_size: int, padding: int = 3) -> torch.Tensor:
        """Merge the patched input into a image with sliding window."""
        steps = int(math.sqrt(x.shape[0] // batch_size))

        idx = 0

        output_list = []
        for j in range(steps):
            output_row_list = []
            for i in range(steps):
                output = x[batch_size * idx : batch_size * (idx + 1)]

                if j != 0:
                    output = output[..., padding:, :]
                if i != 0:
                    output = output[..., :, padding:]
                if j != steps - 1:
                    output = output[..., :-padding, :]
                if i != steps - 1:
                    output = output[..., :, :-padding]

                output_row_list.append(output)
                idx += 1

            output_row = torch.cat(output_row_list, dim=-1)
            output_list.append(output_row)
        output = torch.cat(output_list, dim=-2)
        return output

    def reshape_feature(
        self, embeddings: torch.Tensor, width, height, cls_token_offset=1
    ):
        """Discard class token and reshape 1D feature map to a 2D grid."""
        b, hw, c = embeddings.shape

        # Remove class token.
        if cls_token_offset > 0:
            embeddings = embeddings[:, cls_token_offset:, :]

        # Shape: (batch, height, width, dim) -> (batch, dim, height, width)
        embeddings = embeddings.reshape(b, height, width, c).permute(0, 3, 1, 2)
        return embeddings

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Encode input at multiple resolutions.

        Args:
        ----
            x (torch.Tensor): Input image.

        Returns:
        -------
            Multi resolution encoded features.

        """
        batch_size = x.shape[0]

        # Step 0: create a 3-level image pyramid.
        x0, x1, x2 = self._create_pyramid(x)

        # Step 1: split to create batched overlapped mini-images at the backbone (BeiT/ViT/Dino)
        # resolution.
        # 5x5 @ 384x384 at the highest resolution (1536x1536).
        x0_patches = self.split(x0, overlap_ratio=0.25)
        # 3x3 @ 384x384 at the middle resolution (768x768).
        x1_patches = self.split(x1, overlap_ratio=0.5)
        # 1x1 # 384x384 at the lowest resolution (384x384).
        x2_patches = x2

        # Concatenate all the sliding window patches and form a batch of size (35=5x5+3x3+1x1).
        x_pyramid_patches = torch.cat(
            (x0_patches, x1_patches, x2_patches),
            dim=0,
        )

        # Step 2: Run the backbone (BeiT) model and get the result of large batch size.
        x_pyramid_encodings = self.patch_encoder(x_pyramid_patches)
        x_pyramid_encodings = self.reshape_feature(
            x_pyramid_encodings, self.out_size, self.out_size
        )

        # Step 3: merging.
        # Merge highres latent encoding.
        x_latent0_encodings = self.reshape_feature(
            self.backbone_highres_hook0,
            self.out_size,
            self.out_size,
        )
        x_latent0_features = self.merge(
            x_latent0_encodings[: batch_size * 5 * 5], batch_size=batch_size, padding=3
        )

        x_latent1_encodings = self.reshape_feature(
            self.backbone_highres_hook1,
            self.out_size,
            self.out_size,
        )
        x_latent1_features = self.merge(
            x_latent1_encodings[: batch_size * 5 * 5], batch_size=batch_size, padding=3
        )

        # Split the 35 batch size from pyramid encoding back into 5x5+3x3+1x1.
        x0_encodings, x1_encodings, x2_encodings = torch.split(
            x_pyramid_encodings,
            [len(x0_patches), len(x1_patches), len(x2_patches)],
            dim=0,
        )

        # 96x96 feature maps by merging 5x5 @ 24x24 patches with overlaps.
        x0_features = self.merge(x0_encodings, batch_size=batch_size, padding=3)

        # 48x84 feature maps by merging 3x3 @ 24x24 patches with overlaps.
        x1_features = self.merge(x1_encodings, batch_size=batch_size, padding=6)

        # 24x24 feature maps.
        x2_features = x2_encodings

        # Apply the image encoder model.
        x_global_features = self.image_encoder(x2_patches)
        x_global_features = self.reshape_feature(
            x_global_features, self.out_size, self.out_size
        )

        # Upsample feature maps.
        x_latent0_features = self.upsample_latent0(x_latent0_features)
        x_latent1_features = self.upsample_latent1(x_latent1_features)

        x0_features = self.upsample0(x0_features)
        x1_features = self.upsample1(x1_features)
        x2_features = self.upsample2(x2_features)

        x_global_features = self.upsample_lowres(x_global_features)
        x_global_features = self.fuse_lowres(
            torch.cat((x2_features, x_global_features), dim=1)
        )

        return [
            x_latent0_features,
            x_latent1_features,
            x0_features,
            x1_features,
            x_global_features,
        ]
