"""Pore connectivity and size analysis routines."""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
from scipy import ndimage


CONNECTIVITY_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)


def analyze_connected_components(volume: np.ndarray) -> Dict[str, Any]:
    """Perform basic connected-domain analysis on the pore space."""
    if volume.ndim != 3:
        raise ValueError("volume 必须是三维数组。")

    labeled, num_components = ndimage.label(volume == 1, structure=CONNECTIVITY_STRUCTURE)
    component_sizes = np.bincount(labeled.ravel())
    component_sizes = component_sizes[1:] if component_sizes.size > 1 else np.asarray([], dtype=np.int64)
    largest_component = int(component_sizes.max()) if component_sizes.size else 0

    return {
        "connected_components": int(num_components),
        "largest_component_voxels": largest_component,
        "component_sizes": component_sizes.tolist(),
    }


def estimate_pore_size_statistics(volume: np.ndarray) -> Dict[str, float]:
    """Estimate simple pore-size statistics using equivalent sphere diameters."""
    connectivity = analyze_connected_components(volume)
    component_sizes = np.asarray(connectivity["component_sizes"], dtype=np.float64)
    if component_sizes.size == 0:
        return {"mean_equivalent_diameter": 0.0, "max_equivalent_diameter": 0.0}

    diameters = 2.0 * ((3.0 * component_sizes) / (4.0 * math.pi)) ** (1.0 / 3.0)
    return {
        "mean_equivalent_diameter": float(np.mean(diameters)),
        "max_equivalent_diameter": float(np.max(diameters)),
    }
