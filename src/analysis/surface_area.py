"""Surface area estimation for binary pore-solid volumes."""

from __future__ import annotations

import numpy as np


def estimate_specific_surface_area(volume: np.ndarray) -> float:
    """Approximate specific surface area by counting pore-solid interfaces."""
    if volume.ndim != 3:
        raise ValueError("volume 必须是三维数组。")

    pore = volume.astype(np.uint8)
    interface_count = 0
    for axis in range(3):
        diff = np.abs(np.diff(pore, axis=axis))
        interface_count += int(np.sum(diff))

    pore_voxels = max(int(np.sum(pore)), 1)
    return float(interface_count / pore_voxels)
