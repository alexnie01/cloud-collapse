# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Python simulation illustrating gravitational collapse of a 3D particle cloud into a 2D rotating plane.

## Environment

Managed with **uv** (Python 3.12 — required by `zarr>=3.2`, whose earlier v3 releases are either yanked for data corruption or require 3.11+). The `.venv` is already created.

```bash
# Activate the venv
source .venv/bin/activate

# Install/sync dependencies
uv sync

# Run the simulation (once a CLI entry point exists)
uv run <script>.py
```

## Key Dependencies

| Package | Role |
|---------|------|
| `numpy` / `scipy` | Numerical computation |
| `numba` | JIT-compiled hot loops (N-body integration) |
| `pyvista` / `vtk` | 3D visualization and rendering |
| `matplotlib` | 2D plots and analysis |
| `zarr` | Chunked array storage for trajectory output |
| `cyclopts` | CLI argument parsing |
| `rich` | Terminal output formatting |

## Output Files

Trajectory output is regenerable and excluded from git (`.zarr/`, `.npz`, `.npy`).
