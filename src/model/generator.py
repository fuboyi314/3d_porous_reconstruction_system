"""Demo conditional generator network for porous voxel reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(slots=True)
class GeneratorConfig:
    """Configuration for the demo generator network."""

    parameter_dim: int = 7
    latent_dim: int = 32
    base_channels: int = 16
    base_grid_size: int = 8


class ConditionalVoxelGenerator(nn.Module):
    """Small conditional 3D generator suitable for engineering demos.

    The network is intentionally lightweight and untrained by default so the
    application can run without a real dataset. Users may replace the saved
    weights later with trained parameters using the same architecture.
    """

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        super().__init__()
        self.config = config or GeneratorConfig()
        hidden_dim = self.config.base_channels * self.config.base_grid_size**3

        self.condition_encoder = nn.Sequential(
            nn.Linear(self.config.parameter_dim + self.config.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(self.config.base_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, parameters: Tensor, noise: Tensor, output_size: tuple[int, int, int]) -> Tensor:
        """Generate a dense voxel field in the range 0..1."""
        encoded = self.condition_encoder(torch.cat([parameters, noise], dim=1))
        grid = encoded.view(-1, self.config.base_channels, self.config.base_grid_size, self.config.base_grid_size, self.config.base_grid_size)
        logits = self.decoder(grid)
        resized = F.interpolate(logits, size=output_size, mode="trilinear", align_corners=False)
        return torch.sigmoid(resized)
