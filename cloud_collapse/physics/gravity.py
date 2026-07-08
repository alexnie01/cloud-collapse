from __future__ import annotations

import numpy as np
from numba import njit, prange

__all__ = ["compute_accelerations"]


def compute_accelerations(positions: np.ndarray, masses: np.ndarray, softening: float, g: float = 1.0) -> np.ndarray:
    """All-pairs softened Newtonian gravity. (N,3), (N,) -> (N,3) accelerations.

    This is the sole entry point into gravity for the rest of the codebase.
    Swap the numba kernel below for a JAX/Metal backend later without
    touching any caller.
    """
    return _gravity_kernel(
        np.ascontiguousarray(positions, dtype=np.float32),
        np.ascontiguousarray(masses, dtype=np.float32),
        np.float32(softening),
        np.float32(g),
    )


@njit(parallel=True, fastmath=True, cache=True)
def _gravity_kernel(positions: np.ndarray, masses: np.ndarray, softening: np.float32, g: np.float32) -> np.ndarray:
    n = positions.shape[0]
    accel = np.zeros((n, 3), dtype=np.float32)
    eps2 = softening * softening
    for i in prange(n):
        xi = positions[i, 0]
        yi = positions[i, 1]
        zi = positions[i, 2]
        ax = np.float32(0.0)
        ay = np.float32(0.0)
        az = np.float32(0.0)
        for j in range(n):
            if i == j:
                continue
            dx = positions[j, 0] - xi
            dy = positions[j, 1] - yi
            dz = positions[j, 2] - zi
            dist2 = dx * dx + dy * dy + dz * dz + eps2
            inv_dist3 = np.float32(1.0) / (dist2 * np.sqrt(dist2))
            f = g * masses[j] * inv_dist3
            ax += f * dx
            ay += f * dy
            az += f * dz
        accel[i, 0] = ax
        accel[i, 1] = ay
        accel[i, 2] = az
    return accel
