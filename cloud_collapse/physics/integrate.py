from __future__ import annotations

import numpy as np
from rich.progress import track

from cloud_collapse.initial_conditions import build_initial_state
from cloud_collapse.io.trajectory_store import create_store, write_diagnostics_step, write_frame, write_masses
from cloud_collapse.params import RunParams
from cloud_collapse.physics.collisions import build_cell_list, find_collision_pairs, resolve_collisions
from cloud_collapse.physics.diagnostics import compute_diagnostics
from cloud_collapse.physics.gravity import compute_accelerations

__all__ = ["leapfrog_step", "run_simulation"]


def leapfrog_step(
    positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray, masses: np.ndarray, params: RunParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One kick-drift-kick leapfrog step, then collision resolution on the drifted state.

    Carries `accelerations` across calls so each step costs one gravity
    evaluation (not two) -- material at N up to 50k where gravity is O(N^2).
    """
    half_v = velocities + 0.5 * params.dt * accelerations
    new_positions = positions + params.dt * half_v
    new_accelerations = compute_accelerations(new_positions, masses, params.softening, params.g_constant)
    new_velocities = half_v + 0.5 * params.dt * new_accelerations

    cutoff = 2.0 * params.particle_radius
    cell_list = build_cell_list(new_positions, cell_size=cutoff)
    pairs = find_collision_pairs(new_positions, cutoff, cell_list)
    resolve_collisions(new_positions, new_velocities, masses, pairs, params.restitution, params.v_min_normal)

    return new_positions, new_velocities, new_accelerations


def run_simulation(params: RunParams, store_path: str) -> None:
    rng = np.random.default_rng(params.seed)
    positions, velocities, masses = build_initial_state(params, rng)
    accelerations = compute_accelerations(positions, masses, params.softening, params.g_constant)

    root = create_store(store_path, params, params.n_frames)
    write_masses(root, masses)
    write_frame(root, 0, 0.0, positions, velocities)

    diag = compute_diagnostics(positions, velocities, masses, params.softening, params.g_constant)
    write_diagnostics_step(root, 0, 0.0, diag["kinetic_energy"], diag["potential_energy"], diag["angular_momentum"])

    for step in track(range(1, params.n_steps + 1), description="Simulating"):
        positions, velocities, accelerations = leapfrog_step(positions, velocities, accelerations, masses, params)
        t = step * params.dt

        diag = compute_diagnostics(positions, velocities, masses, params.softening, params.g_constant)
        write_diagnostics_step(
            root, step, t, diag["kinetic_energy"], diag["potential_energy"], diag["angular_momentum"]
        )

        if step % params.frame_stride == 0:
            write_frame(root, step // params.frame_stride, t, positions, velocities)
