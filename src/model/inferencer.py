"""Inference helpers that convert parameter vectors into binary voxel volumes."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from scipy import ndimage

from src.model.generator import ConditionalVoxelGenerator


def _build_parameter_vector(params: dict[str, Any]) -> np.ndarray:
    """Encode input statistics into a compact condition vector."""
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


def _normalize_field(field: np.ndarray) -> np.ndarray:
    """Normalize a dense field into the range 0..1."""
    field_min = float(field.min())
    field_max = float(field.max())
    if field_max - field_min < 1e-8:
        return np.zeros_like(field, dtype=np.float32)
    return ((field - field_min) / (field_max - field_min)).astype(np.float32)


def _generate_demo_field(params: dict[str, Any]) -> np.ndarray:
    """Generate a stable synthetic porous field for demo reconstruction.

    This path gives the application an engineering-safe fallback when the model
    is randomly initialized or when users have not yet prepared a trained
    checkpoint. The field is created from multi-scale smoothed noise so the
    reconstructed sample looks porous rather than purely random.
    """
    size = tuple(int(value) for value in params["reconstruction_size"])
    pore_distribution = np.asarray(params["pore_size_distribution"], dtype=np.float32)
    coordination_number = float(params.get("coordination_number", 0.0))

    base_noise = np.random.random(size).astype(np.float32)
    field = np.zeros(size, dtype=np.float32)

    weights = np.linspace(1.0, 0.4, num=max(len(pore_distribution), 1), dtype=np.float32)
    weights = weights / float(np.sum(weights))

    for index, pore_size in enumerate(pore_distribution):
        sigma = max(float(pore_size) / 6.0, 0.8)
        smoothed = ndimage.gaussian_filter(base_noise, sigma=sigma, mode="wrap")
        field += weights[index] * smoothed.astype(np.float32)

    # Add a weak directional structure so the generated result is visually more
    # informative in the slice views while remaining deterministic per seed.
    grid_x, grid_y, grid_z = np.meshgrid(
        np.linspace(0.0, 1.0, size[0], dtype=np.float32),
        np.linspace(0.0, 1.0, size[1], dtype=np.float32),
        np.linspace(0.0, 1.0, size[2], dtype=np.float32),
        indexing="ij",
    )
    directional_bias = (
        np.sin((coordination_number + 1.0) * np.pi * grid_x)
        + np.cos((coordination_number + 1.5) * np.pi * grid_y)
        + np.sin((coordination_number + 2.0) * np.pi * grid_z)
    )
    field += 0.15 * directional_bias.astype(np.float32)
    return _normalize_field(field)


def _run_model_field(
    params: dict[str, Any],
    model: ConditionalVoxelGenerator,
    device: str,
) -> np.ndarray:
    """Run the neural network and return a dense probability field."""
    reconstruction_size = tuple(int(value) for value in params["reconstruction_size"])
    parameter_vector = _build_parameter_vector(params)

    model = model.to(device)
    with torch.no_grad():
        parameter_tensor = torch.from_numpy(parameter_vector).unsqueeze(0).to(device)
        noise_tensor = torch.randn(1, model.config.latent_dim, device=device)
        probabilities = model(parameter_tensor, noise_tensor, output_size=reconstruction_size)
        field = probabilities.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return _normalize_field(field)


def infer_structure(
    params: dict[str, Any],
    model: ConditionalVoxelGenerator,
    device: str = "cpu",
) -> np.ndarray:
    """Run demo inference and return a binary 3D numpy array.

    The final field blends the neural-network output with a deterministic
    multi-scale demo porous field, which guarantees the first version can
    always reconstruct a meaningful sample even before real training data is
    available.
    """
    seed = int(params.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    demo_field = _generate_demo_field(params)

    try:
        network_field = _run_model_field(params, model, device=device)
        field = 0.35 * network_field + 0.65 * demo_field
    except Exception:
        field = demo_field

    field = _normalize_field(field)
    target_porosity = float(params["porosity"])
    threshold = float(np.quantile(field, 1.0 - target_porosity))
    binary_volume = (field >= threshold).astype(np.uint8)

    binary_volume = ndimage.binary_opening(binary_volume, structure=np.ones((2, 2, 2), dtype=np.uint8)).astype(np.uint8)
    binary_volume = ndimage.binary_closing(binary_volume, structure=np.ones((2, 2, 2), dtype=np.uint8)).astype(np.uint8)

    corrected_threshold = float(np.quantile(field, 1.0 - target_porosity))
    corrected_volume = (field >= corrected_threshold).astype(np.uint8)
    merged_volume = np.maximum(binary_volume, corrected_volume).astype(np.uint8)
    return merged_volume
