"""Porosity and voxel counting analysis routines."""

from __future__ import annotations

from typing import Dict

import numpy as np


def calculate_porosity(volume: np.ndarray) -> float:
    """Calculate actual porosity for a binary pore volume."""
    if volume.ndim != 3:
        raise ValueError("volume 必须是三维数组。")
    return float(np.mean(volume == 1))


def count_phase_voxels(volume: np.ndarray) -> Dict[str, int]:
    """Count pore and solid voxels."""
    pore_voxels = int(np.sum(volume == 1))
    total_voxels = int(volume.size)
    return {
        "pore_voxels": pore_voxels,
        "solid_voxels": total_voxels - pore_voxels,
        "total_voxels": total_voxels,
    }


def extract_typical_slices(volume: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract central slices along x, y, and z axes."""
    if volume.ndim != 3:
        raise ValueError("volume 必须是三维数组。")
    x_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    z_mid = volume.shape[2] // 2
    return {
        "x": volume[x_mid, :, :],
        "y": volume[:, y_mid, :],
        "z": volume[:, :, z_mid],
    }
