from __future__ import annotations

import numpy as np
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

from cloud_collapse.initial_conditions import build_initial_state
from cloud_collapse.io.trajectory_store import create_store, write_diagnostics, write_frame, write_masses
from cloud_collapse.params import RunParams
from cloud_collapse.physics.collisions import build_cell_list, find_collision_pairs, resolve_collisions
from cloud_collapse.physics.diagnostics import (
    angular_momentum,
    kinetic_energy,
    marginal_potential_energy,
    potential_energy,
)
from cloud_collapse.physics.gravity import compute_accelerations

__all__ = ["leapfrog_step", "run_simulation"]


def leapfrog_step(
    positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray, masses: np.ndarray, params: RunParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One kick-drift-kick leapfrog step, then collision resolution on the drifted state.

    Carries `accelerations` across calls so each step costs one gravity
    evaluation (not two) -- material at N up to 50k where gravity is O(N^2).
    Callers pass only the *active* (still-integrated) subset; escaped
    particles never reach this function. Also returns a per-particle boolean
    mask (same indexing as the input subset) of who was in a collision pair
    this step, for visualization highlighting.
    """
    half_v = velocities + 0.5 * params.dt * accelerations
    new_positions = positions + params.dt * half_v
    new_accelerations = compute_accelerations(new_positions, masses, params.softening, params.g_constant)
    new_velocities = half_v + 0.5 * params.dt * new_accelerations

    cutoff = 2.0 * params.particle_radius
    cell_list = build_cell_list(new_positions, cell_size=cutoff)
    pairs = find_collision_pairs(new_positions, cutoff, cell_list)
    resolve_collisions(new_positions, new_velocities, masses, pairs, params.restitution, params.v_min_normal)

    collided = np.zeros(new_positions.shape[0], dtype=bool)
    if pairs.shape[0] > 0:
        collided[pairs[:, 0]] = True
        collided[pairs[:, 1]] = True

    return new_positions, new_velocities, new_accelerations, collided


def _find_new_escapees(positions: np.ndarray, velocities: np.ndarray, params: RunParams) -> np.ndarray:
    """Particles beyond escape_radius moving faster than the (point-mass) escape velocity.

    Both conditions are required so a fast but still-close pericenter passage
    isn't mistaken for an actual ejection.
    """
    r = np.sqrt(np.einsum("ij,ij->i", positions, positions))
    speed = np.sqrt(np.einsum("ij,ij->i", velocities, velocities))
    v_esc = np.sqrt(2.0 * params.g_constant * params.total_mass / np.maximum(r, 1e-12))
    return (r > params.escape_radius) & (speed > v_esc)


def _find_newly_parked(positions: np.ndarray, velocities: np.ndarray, params: RunParams) -> np.ndarray:
    """Coasting particles far/fast enough to be frozen at the origin for good.

    Same escape-velocity gate as `_find_new_escapees`, just at the much larger
    `park_radius` -- only once a particle is unambiguously gone does it get
    collapsed to a point instead of tracked coasting outward forever.
    """
    r = np.sqrt(np.einsum("ij,ij->i", positions, positions))
    speed = np.sqrt(np.einsum("ij,ij->i", velocities, velocities))
    v_esc = np.sqrt(2.0 * params.g_constant * params.total_mass / np.maximum(r, 1e-12))
    return (r > params.park_radius) & (speed > v_esc)


def run_simulation(params: RunParams, store_path: str) -> None:
    rng = np.random.default_rng(params.seed)
    positions, velocities, masses = build_initial_state(params, rng)
    active = np.ones(params.n_particles, dtype=bool)
    parked = np.zeros(params.n_particles, dtype=bool)

    accelerations = np.zeros_like(positions)
    accelerations[active] = compute_accelerations(
        positions[active], masses[active], params.softening, params.g_constant
    )

    root = create_store(store_path, params, params.n_frames)
    write_masses(root, masses)
    no_collisions_yet = np.zeros(params.n_particles, dtype=bool)
    write_frame(root, 0, 0.0, positions, velocities, ~active, no_collisions_yet, parked)

    # Buffered in memory and written once at the end via write_diagnostics -- each
    # diagnostics array is a single Zarr chunk, so per-step incremental writes would
    # force an O(n_steps) re-encode on every call (O(n_steps^2) total).
    n_diag = params.n_steps + 1
    step_times = np.empty(n_diag, dtype=np.float64)
    ke_hist = np.empty(n_diag, dtype=np.float64)
    L_hist = np.empty((n_diag, 3), dtype=np.float64)
    boiled_ke_hist = np.empty(n_diag, dtype=np.float64)
    boiled_L_hist = np.empty((n_diag, 3), dtype=np.float64)

    # KE and L are O(N) and always exact over the whole system (escaped particles still
    # carry their share of the conserved total), so they're recorded every step. PE is
    # O(N^2) -- as expensive as gravity itself -- so it's only evaluated at the same
    # cadence as recorded frames, and only over the active subset. `boiled_*` are running
    # totals moved out of the live computation when a particle escapes (PE, since it's
    # only ever summed over the active set) or gets parked (KE/L, since parking zeroes
    # the position/velocity that the full-array sum below would otherwise keep counting)
    # -- added back in below so kinetic_energy/potential_energy/angular_momentum stay
    # exact whole-system totals throughout the run.
    pe_hist = np.empty(params.n_frames, dtype=np.float64)
    boiled_pe_hist = np.empty(params.n_frames, dtype=np.float64)
    pe_idx = 1

    boiled_ke = 0.0
    boiled_pe = 0.0
    boiled_L = np.zeros(3, dtype=np.float64)

    step_times[0] = 0.0
    ke_hist[0] = kinetic_energy(velocities, masses)
    L_hist[0] = angular_momentum(positions, velocities, masses)
    pe_hist[0] = potential_energy(positions[active], masses[active], params.softening, params.g_constant)
    boiled_ke_hist[0] = boiled_ke
    boiled_L_hist[0] = boiled_L
    boiled_pe_hist[0] = boiled_pe

    progress_columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("active={task.fields[active]} escaped={task.fields[escaped]} parked={task.fields[parked]}"),
        TimeRemainingColumn(),
    )
    # Accumulated (OR'd) across every step since the last recorded frame, then reset --
    # so the viewer can highlight "collided sometime during this displayed interval"
    # rather than only whichever single step happens to land on a recorded frame.
    collided_since_last_frame = np.zeros(params.n_particles, dtype=bool)

    with Progress(*progress_columns) as progress:
        task_id = progress.add_task(
            "Simulating", total=params.n_steps, active=params.n_particles, escaped=0, parked=0
        )

        for step in range(1, params.n_steps + 1):
            active_idx = np.nonzero(active)[0]
            if active_idx.size > 0:
                new_pos, new_vel, new_acc, collided_sub = leapfrog_step(
                    positions[active_idx],
                    velocities[active_idx],
                    accelerations[active_idx],
                    masses[active_idx],
                    params,
                )
                positions[active_idx] = new_pos
                velocities[active_idx] = new_vel
                accelerations[active_idx] = new_acc
                collided_since_last_frame[active_idx] |= collided_sub

            # Coast particles that were *already* escaped before this step -- particles
            # that escape below aren't in this mask yet, so they aren't double-advanced
            # (their position was already updated by leapfrog_step above, this step).
            coasting = ~active & ~parked
            if coasting.any():
                positions[coasting] += params.dt * velocities[coasting]

            if active_idx.size > 0:
                newly_escaped_local = _find_new_escapees(positions[active_idx], velocities[active_idx], params)
                if newly_escaped_local.any():
                    newly_escaped_idx = active_idx[newly_escaped_local]
                    remaining_active_idx = active_idx[~newly_escaped_local]
                    boiled_pe += marginal_potential_energy(
                        positions[newly_escaped_idx],
                        masses[newly_escaped_idx],
                        positions[remaining_active_idx],
                        masses[remaining_active_idx],
                        params.softening,
                        params.g_constant,
                    )
                    active[newly_escaped_idx] = False

            # Re-check coasting (now includes this step's newly escaped, positions all
            # up to date) for anyone who's roamed past park_radius -- freeze them at the
            # origin, banking their KE/L first so the full-array sums below don't lose it.
            coasting_idx = np.nonzero(~active & ~parked)[0]
            if coasting_idx.size > 0:
                newly_parked_local = _find_newly_parked(positions[coasting_idx], velocities[coasting_idx], params)
                if newly_parked_local.any():
                    park_idx = coasting_idx[newly_parked_local]
                    boiled_ke += kinetic_energy(velocities[park_idx], masses[park_idx])
                    boiled_L += angular_momentum(positions[park_idx], velocities[park_idx], masses[park_idx])
                    positions[park_idx] = 0.0
                    velocities[park_idx] = 0.0
                    parked[park_idx] = True

            t = step * params.dt
            active_now = np.nonzero(active)[0]
            step_times[step] = t
            ke_hist[step] = kinetic_energy(velocities, masses) + boiled_ke
            L_hist[step] = angular_momentum(positions, velocities, masses) + boiled_L
            boiled_ke_hist[step] = boiled_ke
            boiled_L_hist[step] = boiled_L

            if step % params.frame_stride == 0:
                write_frame(
                    root,
                    step // params.frame_stride,
                    t,
                    positions,
                    velocities,
                    ~active & ~parked,
                    collided_since_last_frame,
                    parked,
                )
                collided_since_last_frame[:] = False

                pe_live = (
                    potential_energy(positions[active_now], masses[active_now], params.softening, params.g_constant)
                    if active_now.size > 0
                    else 0.0
                )
                pe_hist[pe_idx] = pe_live + boiled_pe
                boiled_pe_hist[pe_idx] = boiled_pe
                pe_idx += 1

            n_parked = int(parked.sum())
            progress.update(
                task_id,
                advance=1,
                active=active_now.size,
                escaped=params.n_particles - active_now.size - n_parked,
                parked=n_parked,
            )

    write_diagnostics(
        root, step_times, ke_hist, L_hist, pe_hist, boiled_ke_hist, boiled_L_hist, boiled_pe_hist
    )
