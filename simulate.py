from __future__ import annotations

import cyclopts
from rich.console import Console

from cloud_collapse.params import RunParams, prompt_run_params
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
    frame_stride: int = 25,
    seed: int = 0,
    out: str = "run.zarr",
    interactive: bool = False,
) -> None:
    """Run the collapse simulation and write a Zarr trajectory store.

    Parameters
    ----------
    n_particles: Particle count (1..50000).
    restitution: Collision restitution for the normal velocity component (0..1).
    v_min_normal: Below this relative normal speed, collisions are perfectly elastic.
    n_steps: Number of integration steps.
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
    frame_stride: Steps between recorded position/velocity frames.
    seed: Random seed.
    out: Output Zarr store path.
    interactive: Prompt for n_particles/restitution/v_min_normal/n_steps instead of using flags.
    """
    if interactive:
        params = prompt_run_params()
    else:
        params = RunParams(
            n_particles=n_particles,
            restitution=restitution,
            v_min_normal=v_min_normal,
            n_steps=n_steps,
            dt=dt,
            cloud_sigma=cloud_sigma,
            cloud_r_max=cloud_r_max,
            thermal_sigma=thermal_sigma,
            omega=(omega_x, omega_y, omega_z),
            softening=softening,
            particle_radius=particle_radius,
            total_mass=total_mass,
            frame_stride=frame_stride,
            seed=seed,
        )

    console.print(
        f"[bold]cloud-collapse[/bold]: N={params.n_particles}, steps={params.n_steps}, "
        f"restitution={params.restitution}, v_min_normal={params.v_min_normal} -> {out}"
    )
    run_simulation(params, out)
    console.print(f"[green]Done.[/green] Trajectory written to {out}")


if __name__ == "__main__":
    app()
