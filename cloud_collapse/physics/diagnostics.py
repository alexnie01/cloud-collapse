from __future__ import annotations

import numpy as np
from numba import njit, prange

__all__ = ["kinetic_energy", "potential_energy", "angular_momentum", "compute_diagnostics"]


def kinetic_energy(velocities: np.ndarray, masses: np.ndarray) -> float:
    speed2 = np.einsum("ij,ij->i", velocities, velocities)
    return float(0.5 * np.dot(masses, speed2))


def potential_energy(positions: np.ndarray, masses: np.ndarray, softening: float, g: float = 1.0) -> float:
    return float(
        _potential_energy_kernel(
            np.ascontiguousarray(positions, dtype=np.float32),
            np.ascontiguousarray(masses, dtype=np.float32),
            np.float32(softening),
            np.float32(g),
        )
    )


@njit(parallel=True, fastmath=True, cache=True)
def _potential_energy_kernel(positions, masses, softening, g):
    n = positions.shape[0]
    eps2 = softening * softening
    partial = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        xi, yi, zi = positions[i, 0], positions[i, 1], positions[i, 2]
        acc = np.float32(0.0)
        for j in range(n):
            if i == j:
                continue
            dx = positions[j, 0] - xi
            dy = positions[j, 1] - yi
            dz = positions[j, 2] - zi
            dist = np.sqrt(dx * dx + dy * dy + dz * dz + eps2)
            acc += masses[j] / dist
        partial[i] = masses[i] * acc
    return -0.5 * g * np.sum(partial)


def angular_momentum(positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
    return (masses[:, None] * np.cross(positions, velocities)).sum(axis=0)


def compute_diagnostics(
    positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray, softening: float, g: float = 1.0
) -> dict:
    return {
        "kinetic_energy": kinetic_energy(velocities, masses),
        "potential_energy": potential_energy(positions, masses, softening, g),
        "angular_momentum": angular_momentum(positions, velocities, masses),
    }
