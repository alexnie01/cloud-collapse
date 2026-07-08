from __future__ import annotations

import numpy as np
import zarr
from zarr.codecs import BloscCname, BloscCodec, BloscShuffle

from cloud_collapse.params import RunParams

__all__ = ["create_store", "write_masses", "write_frame", "write_diagnostics", "open_store", "read_frame"]

_COMPRESSOR = [BloscCodec(cname=BloscCname.zstd, clevel=5, shuffle=BloscShuffle.shuffle)]


def create_store(path: str, params: RunParams, n_frames: int) -> zarr.Group:
    """Create the Zarr v3 store: positions/velocities/times/masses + a diagnostics subgroup.

    Root attrs hold the full RunParams (incl. seed) so a run is reproducible
    from the store alone. positions/velocities are chunked one frame at a
    time for cheap lazy single-frame reads in Stage 2. KE and angular
    momentum are O(N) so they're recorded at full integration-step
    resolution; potential energy is O(N^2) (as expensive as gravity itself),
    so it's recorded at the same coarser cadence as recorded frames, sharing
    the top-level `times` array instead of its own.
    """
    root = zarr.open_group(path, mode="w")
    root.attrs.update(params.to_dict())

    n = params.n_particles
    root.create_array(
        "positions", shape=(n_frames, n, 3), dtype="float32", chunks=(1, n, 3), compressors=_COMPRESSOR
    )
    root.create_array(
        "velocities", shape=(n_frames, n, 3), dtype="float32", chunks=(1, n, 3), compressors=_COMPRESSOR
    )
    root.create_array("times", shape=(n_frames,), dtype="float64", chunks=(n_frames,))
    root.create_array("masses", shape=(n,), dtype="float32", chunks=(n,))
    root.create_array("escaped", shape=(n_frames, n), dtype="bool", chunks=(1, n))
    root.create_array("collided", shape=(n_frames, n), dtype="bool", chunks=(1, n))
    root.create_array("parked", shape=(n_frames, n), dtype="bool", chunks=(1, n))

    n_diag = params.n_steps + 1
    diag = root.create_group("diagnostics")
    diag.create_array("step_times", shape=(n_diag,), dtype="float64", chunks=(n_diag,))
    diag.create_array("kinetic_energy", shape=(n_diag,), dtype="float64", chunks=(n_diag,))
    diag.create_array("angular_momentum", shape=(n_diag, 3), dtype="float64", chunks=(n_diag, 3))
    diag.create_array("potential_energy", shape=(n_frames,), dtype="float64", chunks=(n_frames,))
    # Boiled-off portions of the same three quantities: whenever a particle escapes
    # or gets parked (see park_radius), its contribution moves here instead of just
    # vanishing, so kinetic_energy/potential_energy/angular_momentum above stay exact
    # totals over the whole system rather than only the live/active part.
    diag.create_array("boiled_kinetic_energy", shape=(n_diag,), dtype="float64", chunks=(n_diag,))
    diag.create_array("boiled_angular_momentum", shape=(n_diag, 3), dtype="float64", chunks=(n_diag, 3))
    diag.create_array("boiled_potential_energy", shape=(n_frames,), dtype="float64", chunks=(n_frames,))
    return root


def write_masses(root: zarr.Group, masses: np.ndarray) -> None:
    root["masses"][:] = masses


def write_frame(
    root: zarr.Group,
    frame_idx: int,
    t: float,
    positions: np.ndarray,
    velocities: np.ndarray,
    escaped: np.ndarray,
    collided: np.ndarray,
    parked: np.ndarray,
) -> None:
    root["positions"][frame_idx] = positions
    root["velocities"][frame_idx] = velocities
    root["times"][frame_idx] = t
    root["escaped"][frame_idx] = escaped
    root["collided"][frame_idx] = collided
    root["parked"][frame_idx] = parked


def write_diagnostics(
    root: zarr.Group,
    step_times: np.ndarray,
    kinetic_energy: np.ndarray,
    angular_momentum: np.ndarray,
    potential_energy: np.ndarray,
    boiled_kinetic_energy: np.ndarray,
    boiled_angular_momentum: np.ndarray,
    boiled_potential_energy: np.ndarray,
) -> None:
    """Write the full diagnostic history in one shot.

    Each diagnostics array is a single Zarr chunk (the whole run is small,
    scalar-per-step data), so a per-step incremental write would force a
    decode/re-encode of that entire chunk on every call -- O(n_steps) cost
    per write, O(n_steps^2) total. Buffering in memory during the run and
    writing once here keeps it O(n_steps) overall.

    `step_times`/`kinetic_energy`/`angular_momentum` are full step resolution
    (length n_steps+1); `potential_energy` is at the coarser frame cadence
    (length n_frames) and pairs with the top-level `times` array, not
    `step_times`. The `boiled_*` arrays share the same cadence as their
    live-quantity counterpart and are already included in it (kinetic_energy
    etc. are whole-system totals, not just the active/live part) -- they're
    written separately purely so the boiled-off share is inspectable.
    """
    diag = root["diagnostics"]
    diag["step_times"][:] = step_times
    diag["kinetic_energy"][:] = kinetic_energy
    diag["angular_momentum"][:] = angular_momentum
    diag["potential_energy"][:] = potential_energy
    diag["boiled_kinetic_energy"][:] = boiled_kinetic_energy
    diag["boiled_angular_momentum"][:] = boiled_angular_momentum
    diag["boiled_potential_energy"][:] = boiled_potential_energy


def open_store(path: str) -> zarr.Group:
    return zarr.open_group(path, mode="r")


def read_frame(
    root: zarr.Group, frame_idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        root["positions"][frame_idx],
        root["velocities"][frame_idx],
        root["escaped"][frame_idx],
        root["collided"][frame_idx],
        root["parked"][frame_idx],
    )
