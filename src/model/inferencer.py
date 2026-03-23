"""Inference helpers that convert parameter vectors into binary voxel volumes."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from src.model.generator import ConditionalVoxelGenerator


def _build_parameter_vector(params: dict[str, Any]) -> np.ndarray:
    pore_distribution = np.asarray(params["pore_size_distribution"], dtype=np.float32)
    mean_pore = float(np.mean(pore_distribution))
    std_pore = float(np.std(pore_distribution))
    size_x, size_y, size_z = params["reconstruction_size"]
    vector = np.asarray(
        [
            float(params["porosity"]),
            mean_pore,
            std_pore,
            float(params["specific_surface_area"]),
            float(params["coordination_number"]),
            float(size_x),
            float(size_y + size_z) / 2.0,
        ],
        dtype=np.float32,
    )
    return vector


def infer_structure(
    params: dict[str, Any],
    model: ConditionalVoxelGenerator,
    device: str = "cpu",
) -> np.ndarray:
    """Run demo inference and return a binary 3D numpy array.

    The model output is thresholded to approximate the target porosity so the
    first engineering version remains stable and visually meaningful even with
    randomly initialized weights.
    """
    seed = int(params.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    reconstruction_size = tuple(int(value) for value in params["reconstruction_size"])
    parameter_vector = _build_parameter_vector(params)

    model = model.to(device)
    with torch.no_grad():
        parameter_tensor = torch.from_numpy(parameter_vector).unsqueeze(0).to(device)
        noise_tensor = torch.randn(1, model.config.latent_dim, device=device)
        probabilities = model(parameter_tensor, noise_tensor, output_size=reconstruction_size)
        field = probabilities.squeeze(0).squeeze(0).detach().cpu().numpy()

    target_porosity = float(params["porosity"])
    threshold = float(np.quantile(field, 1.0 - target_porosity))
    binary_volume = (field >= threshold).astype(np.uint8)
    return binary_volume
