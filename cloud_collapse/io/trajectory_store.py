from __future__ import annotations

import numpy as np
import zarr
from zarr.codecs import BloscCname, BloscCodec, BloscShuffle

from cloud_collapse.params import RunParams

__all__ = ["create_store", "write_masses", "write_frame", "write_diagnostics_step", "open_store", "read_frame"]

_COMPRESSOR = [BloscCodec(cname=BloscCname.zstd, clevel=5, shuffle=BloscShuffle.shuffle)]


def create_store(path: str, params: RunParams, n_frames: int) -> zarr.Group:
    """Create the Zarr v3 store: positions/velocities/times/masses + a diagnostics subgroup.

    Root attrs hold the full RunParams (incl. seed) so a run is reproducible
    from the store alone. positions/velocities are chunked one frame at a
    time for cheap lazy single-frame reads in Stage 2; diagnostics are
    recorded at full integration-step resolution, independent of the
    positions/velocities frame stride.
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

    n_diag = params.n_steps + 1
    diag = root.create_group("diagnostics")
    diag.create_array("step_times", shape=(n_diag,), dtype="float64", chunks=(n_diag,))
    diag.create_array("kinetic_energy", shape=(n_diag,), dtype="float64", chunks=(n_diag,))
    diag.create_array("potential_energy", shape=(n_diag,), dtype="float64", chunks=(n_diag,))
    diag.create_array("angular_momentum", shape=(n_diag, 3), dtype="float64", chunks=(n_diag, 3))
    return root


def write_masses(root: zarr.Group, masses: np.ndarray) -> None:
    root["masses"][:] = masses


def write_frame(root: zarr.Group, frame_idx: int, t: float, positions: np.ndarray, velocities: np.ndarray) -> None:
    root["positions"][frame_idx] = positions
    root["velocities"][frame_idx] = velocities
    root["times"][frame_idx] = t


def write_diagnostics_step(root: zarr.Group, step_idx: int, t: float, ke: float, pe: float, L: np.ndarray) -> None:
    diag = root["diagnostics"]
    diag["step_times"][step_idx] = t
    diag["kinetic_energy"][step_idx] = ke
    diag["potential_energy"][step_idx] = pe
    diag["angular_momentum"][step_idx] = L


def open_store(path: str) -> zarr.Group:
    return zarr.open_group(path, mode="r")


def read_frame(root: zarr.Group, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    return root["positions"][frame_idx], root["velocities"][frame_idx]
