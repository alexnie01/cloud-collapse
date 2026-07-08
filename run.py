from __future__ import annotations

import os

import cyclopts
from rich.console import Console

from cloud_collapse.params import RunParams, prompt_run_params
from cloud_collapse.physics.integrate import run_simulation
from visualize import animate, plot_diagnostics

app = cyclopts.App(help="Simulate and then render a cloud-collapse run in one command.")
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
    escape_radius_factor: float = 3.0,
    frame_stride: int = 25,
    seed: int = 0,
    out: str = "run.zarr",
    interactive_params: bool = False,
    movie: str | None = None,
    fps: int = 30,
    interactive_view: bool = False,
    show_diagnostics: bool = False,
) -> None:
    """Run simulate.py's physics, then visualize.py's renderer, back to back.

    By default this writes `out` (the Zarr trajectory) and, next to it, a
    movie file with the same basename (e.g. run.zarr -> run.mp4) -- no
    display required. Pass --interactive-view to open a live PyVista window
    instead of exporting a movie, or --show-diagnostics to also plot
    energy/angular-momentum conservation once the run finishes.

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
    escape_radius_factor: Multiple of cloud_r_max defining "the system" -- unbound particles
        past this radius stop being integrated and coast in a straight line.
    frame_stride: Steps between recorded position/velocity frames.
    seed: Random seed.
    out: Output Zarr store path.
    interactive_params: Prompt for n_particles/restitution/v_min_normal/n_steps instead of flags.
    movie: Movie output path. Defaults to `out` with its extension swapped to .mp4.
    fps: Movie frame rate (ignored with --interactive-view).
    interactive_view: Open a live PyVista window instead of exporting a movie.
    show_diagnostics: Also plot energy/angular-momentum diagnostics once rendering finishes.
    """
    if interactive_params:
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
            escape_radius_factor=escape_radius_factor,
            frame_stride=frame_stride,
            seed=seed,
        )

    console.print(
        f"[bold]cloud-collapse[/bold]: N={params.n_particles}, steps={params.n_steps}, "
        f"restitution={params.restitution}, v_min_normal={params.v_min_normal} -> {out}"
    )
    run_simulation(params, out)
    console.print(f"[green]Simulation done.[/green] Trajectory written to {out}")

    if interactive_view:
        animate(out, fps=fps, export=None)
    else:
        movie_path = movie or os.path.splitext(out)[0] + ".mp4"
        animate(out, fps=fps, export=movie_path)
        console.print(f"[green]Movie written to {movie_path}[/green]")

    if show_diagnostics:
        plot_diagnostics(out)


if __name__ == "__main__":
    app()
