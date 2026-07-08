from __future__ import annotations

import numpy as np

from cloud_collapse.params import RunParams


def sample_positions(n_particles: int, cloud_sigma: float, r_max: float, rng: np.random.Generator) -> np.ndarray:
    """Truncated isotropic 3D Gaussian: sample, reject anything beyond r_max, repeat."""
    positions = np.empty((n_particles, 3), dtype=np.float64)
    filled = 0
    while filled < n_particles:
        n_needed = n_particles - filled
        candidates = rng.normal(scale=cloud_sigma, size=(int(n_needed * 1.5) + 32, 3))
        radii = np.linalg.norm(candidates, axis=1)
        accepted = candidates[radii <= r_max]
        take = min(len(accepted), n_needed)
        positions[filled : filled + take] = accepted[:take]
        filled += take
    return positions.astype(np.float32)


def sample_velocities(
    positions: np.ndarray, thermal_sigma: float, omega: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Zero-mean thermal Gaussian per axis, plus a deterministic solid-body rotation v = Omega x r."""
    thermal = rng.normal(scale=thermal_sigma, size=positions.shape)
    rotation = np.cross(omega, positions.astype(np.float64))
    return (thermal + rotation).astype(np.float32)


def build_initial_state(params: RunParams, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = sample_positions(params.n_particles, params.cloud_sigma, params.cloud_r_max, rng)
    velocities = sample_velocities(positions, params.thermal_sigma, params.omega_vec, rng)
    masses = np.full(params.n_particles, params.particle_mass, dtype=np.float32)
    return positions, velocities, masses
