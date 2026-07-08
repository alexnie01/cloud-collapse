from __future__ import annotations

import os

import cyclopts
from rich.console import Console

from cloud_collapse.params import RunParams, out_name_from_toml, prompt_run_params
from cloud_collapse.paths import data_path
from cloud_collapse.physics.integrate import run_simulation

app = cyclopts.App(help="Stage 1: simulate gravitational collapse of a particle cloud into a rotating disk.")
console = Console()


@app.default
def main(
    n_particles: int = 10_000,
    restitution: float = 0.8,
    v_min_normal: float = 0.01,
    n_steps: int = 5_000,
    *,
    stick_velocity: float = 0.05,
    dt: float = 0.01,
    cloud_sigma: float = 1.0,
    cloud_r_max: float = 4.0,
    thermal_sigma: float = 0.1,
    omega_x: float = 0.0,
    omega_y: float = 0.0,
    omega_z: float = 0.5,
    softening: float = 0.05,
    particle_radius: float = 0.02,
    total_mass: float = 1.0,
    escape_radius_factor: float = 3.0,
    park_radius_factor: float = 20.0,
    frame_stride: int = 25,
    seed: int = 0,
    out: str | None = None,
    interactive: bool = False,
    config: str | None = None,
) -> None:
    """Run the collapse simulation and write a Zarr trajectory store.

    Parameters
    ----------
    config: Path to a TOML run config (see configs/example.toml). When given, it's used
        exclusively -- all other physics flags below are ignored. May also set `out`
        directly in the file; an explicit --out flag still takes precedence over that.
    n_particles: Particle count (1..50000).
    restitution: Collision restitution for the normal velocity component (0..1).
    v_min_normal: Below this relative normal speed, collisions are perfectly elastic.
    n_steps: Number of integration steps.
    stick_velocity: Below this relative normal approach speed, colliding particles merge
        (perfectly inelastic, conserving mass/momentum/angular momentum) instead of
        bouncing at all. Independent of v_min_normal, which only affects non-merging
        bounces. Any angular momentum not explained by the merged body's bulk motion is
        banked as spin (bookkeeping only -- it doesn't feed back into forces or collisions).
    dt: Integration timestep.
    cloud_sigma: Isotropic Gaussian scale of the initial cloud.
    cloud_r_max: Truncation radius for the initial cloud.
    thermal_sigma: Per-axis thermal velocity spread.
    omega_x: X component of the seeded solid-body rotation.
    omega_y: Y component of the seeded solid-body rotation.
    omega_z: Z component of the seeded solid-body rotation.
    softening: Gravitational softening length.
    particle_radius: Particle collision radius.
    total_mass: Total system mass, held fixed regardless of n_particles.
    escape_radius_factor: Multiple of cloud_r_max defining "the system" -- unbound particles
        past this radius stop being integrated and coast in a straight line.
    park_radius_factor: Multiple of cloud_r_max, further out than escape_radius_factor,
        beyond which a coasting particle is frozen at the origin (position/velocity
        zeroed) instead of tracked coasting forever; its KE/PE/L move into the run's
        boiled-off diagnostics so conservation still holds over the whole system.
    frame_stride: Steps between recorded position/velocity frames.
    seed: Random seed.
    out: Run name -- the trajectory is written to data/<out>/<out>.zarr. Defaults to "run",
        or to the config file's own `out` field if --config sets one.
    interactive: Prompt for n_particles/restitution/v_min_normal/n_steps instead of using flags.
    """
    if config is not None:
        params = RunParams.from_toml(config)
        out = out or out_name_from_toml(config)
    elif interactive:
        params = prompt_run_params()
    else:
        params = RunParams(
            n_particles=n_particles,
            restitution=restitution,
            v_min_normal=v_min_normal,
            n_steps=n_steps,
            stick_velocity=stick_velocity,
            dt=dt,
            cloud_sigma=cloud_sigma,
            cloud_r_max=cloud_r_max,
            thermal_sigma=thermal_sigma,
            omega=(omega_x, omega_y, omega_z),
            softening=softening,
            particle_radius=particle_radius,
            total_mass=total_mass,
            escape_radius_factor=escape_radius_factor,
            park_radius_factor=park_radius_factor,
            frame_stride=frame_stride,
            seed=seed,
        )

    store_path = data_path(out or "run")
    os.makedirs(os.path.dirname(store_path), exist_ok=True)

    console.print(
        f"[bold]cloud-collapse[/bold]: N={params.n_particles}, steps={params.n_steps}, "
        f"restitution={params.restitution}, v_min_normal={params.v_min_normal} -> {store_path}"
    )
    run_simulation(params, store_path)
    console.print(f"[green]Done.[/green] Trajectory written to {store_path}")


if __name__ == "__main__":
    app()
