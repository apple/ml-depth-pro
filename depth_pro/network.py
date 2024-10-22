from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    Normalize,
)

from ._component import vit
from ._component.decoder import Decoder
from ._component.encoder import Encoder
from ._component.fov import FovNetwork


class Config:
    checkpoint: Path

    decoder_features: int
    output_shape: tuple[int, int]

    patch_encoder_preset: vit.Preset
    patch_encoder_config: vit.Config

    image_encoder_preset: vit.Preset
    image_encoder_config: vit.Config

    fov_encoder_preset: vit.Preset | None
    fov_encoder_config: vit.Config | None

    def __init__(
        self,
        checkpoint: Path,
        decoder_features: int = 256,
        output_shape: tuple[int, int] = (32, 1),
        patch_encoder_preset: vit.Preset = 'dinov2l16_384',
        image_encoder_preset: vit.Preset = 'dinov2l16_384',
        fov_encoder_preset: vit.Preset | None = 'dinov2l16_384',
    ) -> None:
        self.checkpoint = checkpoint
        self.decoder_features = decoder_features
        self.output_shape = output_shape

        self.patch_encoder_preset = patch_encoder_preset
        self.path_encoder_config = vit.PRESETS[patch_encoder_preset]

        self.image_encoder_preset = image_encoder_preset
        self.image_encoder_config = vit.PRESETS[image_encoder_preset]

        self.fov_encoder_preset = fov_encoder_preset
        self.fov_encoder_config = (
            vit.PRESETS[fov_encoder_preset] if fov_encoder_preset is not None else None
        )


@dataclass
class Result:
    focal_length_px: float
    depth: torch.Tensor


class DepthPro(nn.Module):
    """DepthPro network."""

    config: Config
    device: torch.device
    dtype: torch.dtype

    input_transformation: Compose

    encoder: Encoder
    decoder: Decoder
    fov: FovNetwork | None

    def __init__(
        self,
        config: Config,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Initialize DepthPro.

        Args:
        ----
            encoder: The `Encoder` backbone.
            decoder: The `Decoder` decoder.
            last_dims: The dimension for the last convolution layers.
            use_fov_head: Whether to use the field-of-view head.
            fov_encoder: A separate encoder for the field of view.

        """
        super().__init__()

        self.config = config
        self.device = device
        self.dtype = dtype

        self.input_transformation = Compose(
            [
                ConvertImageDtype(dtype),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        output_shape = config.output_shape

        patch_encoder_config = config.path_encoder_config
        fov_encoder_config = config.fov_encoder_config

        patch_encoder = vit.create(
            preset=config.patch_encoder_preset,
            use_pretrained=False,
        )

        image_encoder = vit.create(
            preset=config.image_encoder_preset,
            use_pretrained=False,
        )

        fov_encoder = (
            vit.create(
                preset=config.fov_encoder_preset,
                use_pretrained=False,
            )
            if config.fov_encoder_preset is not None
            else None
        )

        dims_encoder = patch_encoder_config.encoder_feature_dims
        hook_block_ids = patch_encoder_config.encoder_feature_layer_ids

        encoder = Encoder(
            dims_encoder=dims_encoder,
            patch_encoder=patch_encoder,
            image_encoder=image_encoder,
            hook_block_ids=hook_block_ids,
            decoder_features=config.decoder_features,
        )

        decoder = Decoder(
            dims_encoder=[config.decoder_features] + list(encoder.dims_encoder),
            dim_decoder=config.decoder_features,
        )

        dim_decoder = decoder.dim_decoder

        head = nn.Sequential(
            nn.Conv2d(
                dim_decoder,
                dim_decoder // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ConvTranspose2d(
                in_channels=dim_decoder // 2,
                out_channels=dim_decoder // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.Conv2d(
                dim_decoder // 2,
                output_shape[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_shape[0],
                output_shape[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
        )
        head[4].bias.data.fill_(0)

        fov = (
            FovNetwork(
                num_features=dim_decoder,
                fov_encoder=fov_encoder,
            ).to(device)
            if fov_encoder_config is not None
            else None
        )

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.head = head.to(device)
        self.fov = fov

        if config.checkpoint is not None:
            state_dict = torch.load(config.checkpoint, map_location='cpu')
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict=state_dict, strict=True
            )

            if len(unexpected_keys) != 0:
                raise KeyError(
                    f'Depth Pro checkpoint contains unexpected keys: {unexpected_keys}'
                )

            # fc_norm is only for the classification head,
            # which we would not use. We only use the encoding.
            missing_keys = [key for key in missing_keys if 'fc_norm' not in key]
            if len(missing_keys) != 0:
                raise KeyError(f'Depth Pro checkpoint is missing keys: {missing_keys}')

        if dtype == torch.half:
            self.half()

    @property
    def input_image_size(self) -> int:
        """Return the internal image size of the network."""
        return self.encoder.img_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Decode by projection and fusion of multi-resolution encodings.

        Args:
        ----
            x (torch.Tensor): Input image.

        Returns:
        -------
            The canonical inverse depth map [m] and the optional estimated field of view [deg].

        """
        _, _, H, W = x.shape
        assert H == self.input_image_size and W == self.input_image_size

        encodings = self.encoder(x)
        features, features_0 = self.decoder(encodings)
        canonical_inverse_depth = self.head(features)

        fov_deg = None
        if hasattr(self, 'fov'):
            fov_deg = self.fov.forward(x, features_0.detach())

        return canonical_inverse_depth, fov_deg

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        focal_length_px: float | None = None,
        interpolation_mode: str = 'bilinear',
    ) -> Result:
        """Infer depth and fov for a given image.

        If the image is not at network resolution, it is resized to 1536x1536 and
        the estimated depth is resized to the original image resolution.
        Note: if the focal length is given, the estimated value is ignored and the provided
        focal length is use to generate the metric depth values.

        Args:
        ----
            x (torch.Tensor): Input image
            focal_length_px (torch.Tensor): Optional focal length in pixels corresponding to `x`.
            interpolation_mode (str): Interpolation function for downsampling/upsampling.

        Returns:
        ----
            Tensor dictionary (torch.Tensor): depth [m], focallength [pixels].
        """

        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        if focal_length_px is None and self.fov is None:
            raise ValueError(
                'Focal length was not specified. It cannot be infered because FOV head is off.'
            )

        _, _, height, width = x.shape
        resize = height != self.input_image_size or width != self.input_image_size

        if resize:
            x = nn.functional.interpolate(
                x,
                size=(self.input_image_size, self.input_image_size),
                mode=interpolation_mode,
                align_corners=False,
            )

        canonical_inverse_depth, fov_angle_degrees = self.forward(x)

        match focal_length_px, fov_angle_degrees:
            case None, None:
                assert False, 'unreachable'

            case float(value), _:
                f_px = torch.tensor(value)

            case None, angle:
                fov_angle_degrees = angle.to(torch.float)
                f_px = torch.squeeze(0.5 * width / torch.tan(0.5 * torch.deg2rad(angle)))

            case _, _:
                assert False, 'unreachable'

        inverse_depth = canonical_inverse_depth * (width / f_px)

        if resize:
            inverse_depth = nn.functional.interpolate(
                inverse_depth,
                size=(height, width),
                mode=interpolation_mode,
                align_corners=False,
            )

        depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)

        return Result(f_px.item(), depth.squeeze())
