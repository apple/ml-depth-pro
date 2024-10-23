import math

import torch
import torch.nn as nn
from torch.nn import functional


def _project_upsample_block(
    input: int,
    output: int,
    upsampling_layers: int,
    intermediate: int | None = None,
) -> nn.Module:
    """Build a block responsible for projection and upsampling.

    Args:
    ----
        input: Number of input features.
        output: Number of output features.
        n_upsample layers: Numer of upsampling layers.
        intermediate: Numer of features between the first and second block in sequence.
    """

    if intermediate is None:
        intermediate = output

    # Projection.
    initial = [
        nn.Conv2d(
            in_channels=input,
            out_channels=intermediate,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        ),
        nn.ConvTranspose2d(
            in_channels=intermediate,
            out_channels=output,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
        ),
    ]

    upsampling = [
        nn.ConvTranspose2d(
            in_channels=output,
            out_channels=output,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
        )
        for i in range(1, upsampling_layers)
    ]

    return nn.Sequential(*initial, *upsampling)


class Encoder(nn.Module):
    """DepthPro Encoder.

    An encoder aimed at creating multi-resolution encodings from Vision Transformers.
    """

    decoder_features: int
    output_size: int
    stage_dimensions: list[int]
    hook_block_ids: list[int]

    patch_encoder: nn.Module
    image_encoder: nn.Module

    upsample_latent0: nn.Module
    upsample_latent1: nn.Module

    upsample0: nn.Module
    upsample1: nn.Module
    upsample2: nn.Module

    upsample_lowres: nn.ConvTranspose2d
    fuse_lowres: nn.Conv2d

    def __init__(
        self,
        decoder_features: int,
        stage_dimensions: list[int],
        hook_block_ids: list[int],
        patch_encoder: nn.Module,
        image_encoder: nn.Module,
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
            stage_dimensions: Dimensions of the encoder at different layers.
            patch_encoder: Backbone used for patches.
            image_encoder: Backbone used for global image encoder.
            hook_block_ids: Hooks to obtain intermediate features for the patch encoder model.
            decoder_features: Number of feature output in the decoder.

        """
        super().__init__()

        self.stage_dimensions = stage_dimensions
        self.hook_block_ids = hook_block_ids

        patch_encoder_embed_dim = patch_encoder.embed_dim
        image_encoder_embed_dim = image_encoder.embed_dim

        self.output_size = int(
            patch_encoder.patch_embed.img_size[0]
            // patch_encoder.patch_embed.patch_size[0]
        )

        self.upsample_latent0 = _project_upsample_block(
            input=patch_encoder_embed_dim,
            intermediate=stage_dimensions[0],
            output=decoder_features,
            upsampling_layers=3,
        )

        self.upsample_latent1 = _project_upsample_block(
            input=patch_encoder_embed_dim,
            output=stage_dimensions[0],
            upsampling_layers=2,
        )

        self.upsample0 = _project_upsample_block(
            input=patch_encoder_embed_dim,
            output=stage_dimensions[1],
            upsampling_layers=1,
        )

        self.upsample1 = _project_upsample_block(
            input=patch_encoder_embed_dim,
            output=stage_dimensions[2],
            upsampling_layers=1,
        )

        self.upsample2 = _project_upsample_block(
            input=patch_encoder_embed_dim,
            output=stage_dimensions[3],
            upsampling_layers=1,
        )

        self.upsample_lowres = nn.ConvTranspose2d(
            in_channels=image_encoder_embed_dim,
            out_channels=stage_dimensions[3],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )

        self.fuse_lowres = nn.Conv2d(
            in_channels=2 * stage_dimensions[3],
            out_channels=stage_dimensions[3],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Obtain intermediate outputs of the blocks.
        patch_encoder.blocks[hook_block_ids[0]].register_forward_hook(self.hook0)
        patch_encoder.blocks[hook_block_ids[1]].register_forward_hook(self.hook1)

        self.patch_encoder = patch_encoder
        self.image_encoder = image_encoder

    def hook0(self, model, input, output):
        self.backbone_highres_hook0 = output

    def hook1(self, model, input, output):
        self.backbone_highres_hook1 = output

    @property
    def image_size(self) -> int:
        """Return the full image size of the SPN network."""
        return self.patch_encoder.patch_embed.img_size[0] * 4

    def image_pyramid(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a 3-level image pyramid."""
        # Original resolution: 1536 by default.
        x0 = x

        # Middle resolution: 768 by default.
        x1 = functional.interpolate(
            x,
            size=None,
            scale_factor=0.5,
            mode='bilinear',
            align_corners=False,
        )

        # Low resolution: 384 by default, corresponding to the backbone resolution.
        x2 = functional.interpolate(
            x,
            size=None,
            scale_factor=0.25,
            mode='bilinear',
            align_corners=False,
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

        flat_index = 0

        outputs: list[torch.Tensor] = []

        for j in range(steps):
            output_rows: list[torch.Tensor] = []

            for i in range(steps):
                output = x[batch_size * flat_index : batch_size * (flat_index + 1)]

                if j != 0:
                    output = output[..., padding:, :]
                if i != 0:
                    output = output[..., :, padding:]
                if j != steps - 1:
                    output = output[..., :-padding, :]
                if i != steps - 1:
                    output = output[..., :, :-padding]

                output_rows.append(output)
                flat_index += 1

            output_row = torch.cat(output_rows, dim=-1)
            outputs.append(output_row)

        return torch.cat(outputs, dim=-2)

    def reshape_feature(
        self,
        embeddings: torch.Tensor,
        width,
        height,
        cls_token_offset=1,
    ) -> torch.Tensor:
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
        x0, x1, x2 = self.image_pyramid(x)

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
            x_pyramid_encodings, self.output_size, self.output_size
        )

        # Step 3: merging.
        # Merge highres latent encoding.
        x_latent0_encodings = self.reshape_feature(
            self.backbone_highres_hook0,
            self.output_size,
            self.output_size,
        )
        x_latent0_features = self.merge(
            x_latent0_encodings[: batch_size * 5 * 5], batch_size=batch_size, padding=3
        )

        x_latent1_encodings = self.reshape_feature(
            self.backbone_highres_hook1,
            self.output_size,
            self.output_size,
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
            x_global_features, self.output_size, self.output_size
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
