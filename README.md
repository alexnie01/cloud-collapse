# cloud-collapse

A particle-based simulation of a 3D gas/dust cloud gravitationally collapsing into a
rotating disk -- N-body gravity, collisional accretion, and escape handling, with full
energy/angular-momentum conservation bookkeeping.

## What it does

Start with a cloud of particles in a truncated Gaussian distribution, given a small
solid-body spin and thermal velocity spread. Under mutual gravity and inelastic
collisions, the cloud collapses; collisions bleed off the energy that would otherwise
keep it puffed up (a stand-in for radiative cooling), while angular momentum is
conserved, so it settles into a flattened, rotating disk.

- **Gravity**: all-pairs softened Newtonian N-body, numba-parallel.
- **Collisions**: uniform-grid neighbor search each step; pairs bounce with a configurable
  restitution, or merge (perfectly inelastic sticking/accretion) below a relative-velocity
  threshold, conserving mass, momentum, and angular momentum through every merge.
- **Escape handling**: particles that become unbound and leave the system stop being
  integrated (an ejected particle doesn't need gravity computed against it forever), and
  are frozen once they're unambiguously gone for good.
- **Conservation diagnostics**: kinetic/potential energy and angular momentum are tracked
  as exact whole-system totals throughout a run, even as particles escape, freeze, or
  merge -- so you can verify the simulation is physically sound, not just that it looks
  plausible.

## Setup

Managed with [uv](https://docs.astral.sh/uv/) (Python 3.12+):

```bash
uv sync
```

## Usage

```bash
# Simulate and render a run in one command (writes data/<name>/<name>.zarr and
# outputs/<name>/<name>.mp4)
uv run src/run.py --n-particles 10000 --n-steps 5000 --out my_run

# ...or run the stages separately
uv run src/simulate.py --n-particles 10000 --n-steps 5000 --out my_run
uv run src/visualize.py show my_run          # interactive PyVista viewer
uv run src/visualize.py fallback my_run      # matplotlib viewer (no GPU/VTK needed)
uv run src/visualize.py diagnostics my_run   # plot energy/angular-momentum conservation
```

Pass `--help` to any script for the full list of physical parameters (particle count,
restitution, collision stickiness, cloud shape, rotation, softening, escape radii, etc.).

See `CLAUDE.md` for a closer look at the physics model and project layout.
