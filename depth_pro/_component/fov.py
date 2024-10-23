import torch
from torch import nn
from torch.nn import functional


class FovNetwork(nn.Module):
    """Field of View estimation network."""

    features: int

    encoder: nn.Sequential | None
    downsample: nn.Sequential | None
    head: nn.Sequential

    def __init__(
        self,
        features: int,
        fov_encoder: nn.Module | None = None,
    ):
        """Initialize the Field of View estimation block.

        Args:
        ----
            features: Number of features used.
            fov_encoder: Optional encoder to bring additional network capacity.

        """
        super().__init__()

        self.features = features

        fov_head0 = [
            nn.Conv2d(
                features,
                features // 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),  # 128 x 24 x 24
            nn.ReLU(inplace=True),
        ]

        fov_head = [
            nn.Conv2d(
                features // 2,
                features // 4,
                kernel_size=3,
                stride=2,
                padding=1,
            ),  # 64 x 12 x 12
            nn.ReLU(inplace=True),
            nn.Conv2d(
                features // 4,
                features // 8,
                kernel_size=3,
                stride=2,
                padding=1,
            ),  # 32 x 6 x 6
            nn.ReLU(inplace=True),
            nn.Conv2d(
                features // 8,
                1,
                kernel_size=6,
                stride=1,
                padding=0,
            ),
        ]

        if fov_encoder is not None:
            self.encoder = nn.Sequential(
                fov_encoder,
                nn.Linear(fov_encoder.embed_dim, features // 2),
            )

            self.downsample = nn.Sequential(*fov_head0)

        else:
            self.encoder = None
            self.downsample = None

            fov_head = fov_head0 + fov_head

        self.head = nn.Sequential(*fov_head)

    def forward(
        self,
        x: torch.Tensor,
        low_resolution_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Forward the fov network.

        Args:
        ----
            x (torch.Tensor): Input image.
            lowres_feature (torch.Tensor): Low resolution feature.

        Returns:
        ----
            The field of view tensor.
        """

        if self.encoder is None or self.downsample is None:
            return self.head(low_resolution_feature)

        low_resolution_feature = self.downsample(low_resolution_feature)

        x = functional.interpolate(
            x,
            size=None,
            scale_factor=0.25,
            mode='bilinear',
            align_corners=False,
        )
        x = self.encoder(x)[:, 1:].permute(0, 2, 1)
        x = x.reshape_as(low_resolution_feature) + low_resolution_feature

        return self.head(x)
