from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

__all__ = ["CellList", "build_cell_list", "find_collision_pairs", "resolve_collisions"]


@dataclass
class CellList:
    cell_coords: np.ndarray  # (N, 3) int64 grid coords per particle, indexed by particle id
    unique_cells: np.ndarray  # (k,) int64 sorted flat ids of *occupied* cells only
    cell_start: np.ndarray  # (k + 1,) int64 CSR offsets into sorted_ids for unique_cells
    sorted_ids: np.ndarray  # (N,) int64 particle ids sorted by flat cell index
    dims: np.ndarray  # (3,) int64 grid dimensions (bounds-checking + flat-index arithmetic only)
    cell_size: float
    box_min: np.ndarray  # (3,) float64


# Guards only against genuine divergence (a bounding box blown up across a fixed cell
# size), not against ordinary sparse configurations -- occupied-cell storage below
# means memory scales with N regardless of how large dims gets, so this only needs to
# rule out int64 overflow in the flat-index arithmetic (cx + cy*nx + cz*nx*ny).
_MAX_DIMS_PRODUCT = 10**15


def build_cell_list(positions: np.ndarray, cell_size: float) -> CellList:
    """Bucket particles into a uniform grid sized to the current bounding box.

    The cloud is unbounded (it collapses, not confined to a box), so bounds
    are recomputed from the current positions every call. Only cells that
    actually contain a particle are stored (CSR over unique flat cell ids),
    so memory scales with N, not with bounding-box volume / cell_size**3 --
    the interaction radius is normally far smaller than the cloud extent, so
    the vast majority of grid cells are empty.
    """
    positions = np.asarray(positions, dtype=np.float64)
    box_min = positions.min(axis=0) - cell_size
    box_max = positions.max(axis=0) + cell_size
    dims = np.maximum(np.ceil((box_max - box_min) / cell_size).astype(np.int64), 1)
    dims_product = int(dims[0]) * int(dims[1]) * int(dims[2])
    if dims_product > _MAX_DIMS_PRODUCT:
        raise RuntimeError(
            f"Cell grid dims={tuple(dims)} (cell_size={cell_size}, bounding box extent="
            f"{tuple(box_max - box_min)}) are far beyond a sane range. This means positions "
            "have diverged (numerical instability) -- check dt/softening, not this cap."
        )
    cell_coords = np.floor((positions - box_min) / cell_size).astype(np.int64)
    cell_coords = np.clip(cell_coords, 0, dims - 1)
    flat = cell_coords[:, 0] + cell_coords[:, 1] * dims[0] + cell_coords[:, 2] * dims[0] * dims[1]
    order = np.argsort(flat, kind="stable")
    sorted_flat = flat[order]
    unique_cells, start_idx = np.unique(sorted_flat, return_index=True)
    cell_start = np.empty(len(unique_cells) + 1, dtype=np.int64)
    cell_start[:-1] = start_idx
    cell_start[-1] = len(sorted_flat)
    return CellList(
        cell_coords=cell_coords.astype(np.int64),
        unique_cells=unique_cells.astype(np.int64),
        cell_start=cell_start,
        sorted_ids=order.astype(np.int64),
        dims=dims,
        cell_size=cell_size,
        box_min=box_min,
    )


@njit(inline="always")
def _lookup_cell(ncell, unique_cells, cell_start):
    n_unique = unique_cells.shape[0]
    pos = np.searchsorted(unique_cells, ncell)
    if pos >= n_unique or unique_cells[pos] != ncell:
        return -1, -1
    return cell_start[pos], cell_start[pos + 1]


@njit(parallel=True, cache=True)
def _count_pairs(positions, cutoff2, cell_coords, unique_cells, cell_start, sorted_ids, dims):
    n = positions.shape[0]
    nx, ny, nz = dims[0], dims[1], dims[2]
    counts = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        cx, cy, cz = cell_coords[i, 0], cell_coords[i, 1], cell_coords[i, 2]
        xi, yi, zi = positions[i, 0], positions[i, 1], positions[i, 2]
        cnt = 0
        for ddz in range(-1, 2):
            ncz = cz + ddz
            if ncz < 0 or ncz >= nz:
                continue
            for ddy in range(-1, 2):
                ncy = cy + ddy
                if ncy < 0 or ncy >= ny:
                    continue
                for ddx in range(-1, 2):
                    ncx = cx + ddx
                    if ncx < 0 or ncx >= nx:
                        continue
                    ncell = ncx + ncy * nx + ncz * nx * ny
                    start, end = _lookup_cell(ncell, unique_cells, cell_start)
                    if start < 0:
                        continue
                    for k in range(start, end):
                        j = sorted_ids[k]
                        if j <= i:
                            continue
                        dx = positions[j, 0] - xi
                        dy = positions[j, 1] - yi
                        dz = positions[j, 2] - zi
                        dist2 = dx * dx + dy * dy + dz * dz
                        if dist2 < cutoff2:
                            cnt += 1
        counts[i] = cnt
    return counts


@njit(parallel=True, cache=True)
def _fill_pairs(positions, cutoff2, cell_coords, unique_cells, cell_start, sorted_ids, dims, offsets, pairs):
    n = positions.shape[0]
    nx, ny, nz = dims[0], dims[1], dims[2]
    for i in prange(n):
        cx, cy, cz = cell_coords[i, 0], cell_coords[i, 1], cell_coords[i, 2]
        xi, yi, zi = positions[i, 0], positions[i, 1], positions[i, 2]
        write = offsets[i]
        for ddz in range(-1, 2):
            ncz = cz + ddz
            if ncz < 0 or ncz >= nz:
                continue
            for ddy in range(-1, 2):
                ncy = cy + ddy
                if ncy < 0 or ncy >= ny:
                    continue
                for ddx in range(-1, 2):
                    ncx = cx + ddx
                    if ncx < 0 or ncx >= nx:
                        continue
                    ncell = ncx + ncy * nx + ncz * nx * ny
                    start, end = _lookup_cell(ncell, unique_cells, cell_start)
                    if start < 0:
                        continue
                    for k in range(start, end):
                        j = sorted_ids[k]
                        if j <= i:
                            continue
                        dx = positions[j, 0] - xi
                        dy = positions[j, 1] - yi
                        dz = positions[j, 2] - zi
                        dist2 = dx * dx + dy * dy + dz * dz
                        if dist2 < cutoff2:
                            pairs[write, 0] = i
                            pairs[write, 1] = j
                            write += 1


def find_collision_pairs(positions: np.ndarray, cutoff: float, cell_list: CellList) -> np.ndarray:
    """Unordered collision pairs (center distance < cutoff), found via the cell list.

    Neighbor search is read-only and parallelized (prange). The output is
    dynamically sized, so it's built with a safe two-pass count-then-fill:
    each particle i gets a disjoint, pre-computed write range in `pairs`,
    so pass 2 has no cross-thread races even though it also runs in prange.
    """
    positions32 = np.ascontiguousarray(positions, dtype=np.float32)
    cutoff2 = np.float32(cutoff) ** 2
    counts = _count_pairs(
        positions32,
        cutoff2,
        cell_list.cell_coords,
        cell_list.unique_cells,
        cell_list.cell_start,
        cell_list.sorted_ids,
        cell_list.dims,
    )
    total = int(counts.sum())
    if total == 0:
        return np.empty((0, 2), dtype=np.int64)
    offsets = np.empty_like(counts)
    offsets[0] = 0
    np.cumsum(counts[:-1], out=offsets[1:])
    pairs = np.empty((total, 2), dtype=np.int64)
    _fill_pairs(
        positions32,
        cutoff2,
        cell_list.cell_coords,
        cell_list.unique_cells,
        cell_list.cell_start,
        cell_list.sorted_ids,
        cell_list.dims,
        offsets,
        pairs,
    )
    return pairs


@njit(cache=True)
def resolve_collisions(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    pairs: np.ndarray,
    restitution: float,
    v_min_normal: float,
) -> None:
    """Resolve each pair's normal-component velocity in place.

    Below v_min_normal relative normal speed, the collision is treated as
    perfectly elastic (restitution forced to 1) regardless of the configured
    restitution. Deliberately sequential: pairs can share a particle, so
    parallel updates would race on that particle's velocity.
    """
    for k in range(pairs.shape[0]):
        i = pairs[k, 0]
        j = pairs[k, 1]
        dx = positions[j, 0] - positions[i, 0]
        dy = positions[j, 1] - positions[i, 1]
        dz = positions[j, 2] - positions[i, 2]
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 1e-12:
            continue
        nx, ny, nz = dx / dist, dy / dist, dz / dist
        vrx = velocities[i, 0] - velocities[j, 0]
        vry = velocities[i, 1] - velocities[j, 1]
        vrz = velocities[i, 2] - velocities[j, 2]
        vn = vrx * nx + vry * ny + vrz * nz
        if vn <= 0.0:
            continue  # separating along the normal, no response needed
        e_eff = 1.0 if abs(vn) < v_min_normal else restitution
        inv_mi = 1.0 / masses[i]
        inv_mj = 1.0 / masses[j]
        jn = -(1.0 + e_eff) * vn / (inv_mi + inv_mj)
        velocities[i, 0] += jn * inv_mi * nx
        velocities[i, 1] += jn * inv_mi * ny
        velocities[i, 2] += jn * inv_mi * nz
        velocities[j, 0] -= jn * inv_mj * nx
        velocities[j, 1] -= jn * inv_mj * ny
        velocities[j, 2] -= jn * inv_mj * nz
