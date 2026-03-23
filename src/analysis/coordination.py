"""Coordination-number estimation for porous structures."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


NEIGHBOR_KERNEL = np.zeros((3, 3, 3), dtype=np.int8)
NEIGHBOR_KERNEL[1, 1, 0] = 1
NEIGHBOR_KERNEL[1, 1, 2] = 1
NEIGHBOR_KERNEL[1, 0, 1] = 1
NEIGHBOR_KERNEL[1, 2, 1] = 1
NEIGHBOR_KERNEL[0, 1, 1] = 1
NEIGHBOR_KERNEL[2, 1, 1] = 1


def estimate_coordination_number(volume: np.ndarray) -> float:
    """Estimate average 6-neighbor connectivity within the pore phase."""
    if volume.ndim != 3:
        raise ValueError("volume 必须是三维数组。")

    pore = (volume == 1).astype(np.int8)
    if int(np.sum(pore)) == 0:
        return 0.0

    neighbor_counts = ndimage.convolve(pore, NEIGHBOR_KERNEL, mode="constant", cval=0)
    valid_counts = neighbor_counts[pore == 1]
    return float(np.mean(valid_counts))
