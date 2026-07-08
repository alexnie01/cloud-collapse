from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class RunParams:
    """Full configuration for one simulation run. Reproducible from this + seed alone."""

    n_particles: int
    restitution: float
    v_min_normal: float
    n_steps: int
    dt: float = 0.01
    cloud_sigma: float = 1.0
    cloud_r_max: float = 4.0
    thermal_sigma: float = 0.1
    omega: tuple[float, float, float] = (0.0, 0.0, 0.5)
    softening: float = 0.05
    particle_radius: float = 0.02
    g_constant: float = 1.0
    total_mass: float = 1.0
    frame_stride: int = 25
    seed: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.n_particles <= 50_000:
            raise ValueError(f"n_particles must be in [1, 50000], got {self.n_particles}")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError(f"restitution must be in [0, 1], got {self.restitution}")
        if self.v_min_normal < 0.0:
            raise ValueError(f"v_min_normal must be >= 0, got {self.v_min_normal}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")
        if self.frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {self.frame_stride}")
        if self.softening <= 0.0:
            raise ValueError(f"softening must be > 0, got {self.softening}")
        if self.particle_radius <= 0.0:
            raise ValueError(f"particle_radius must be > 0, got {self.particle_radius}")
        if self.total_mass <= 0.0:
            raise ValueError(f"total_mass must be > 0, got {self.total_mass}")

    @property
    def n_frames(self) -> int:
        """Recorded position/velocity frames, including the initial state at step 0."""
        return self.n_steps // self.frame_stride + 1

    @property
    def omega_vec(self) -> np.ndarray:
        return np.asarray(self.omega, dtype=np.float64)

    @property
    def particle_mass(self) -> float:
        """Per-particle mass, holding total system mass fixed regardless of N.

        Keeps gravity strength (and thus stability/close-encounter rate) tied
        to the physical system, not to particle count -- more particles means
        finer resolution of the same cloud, not a heavier one.
        """
        return self.total_mass / self.n_particles

    def to_dict(self) -> dict:
        return asdict(self)


def prompt_run_params() -> RunParams:
    """Interactively collect the run-defining parameters via rich prompts.

    Only the parameters the user must decide per-run are asked here; the rest
    of RunParams keeps its tuned defaults and is reachable via CLI flags.
    """
    from rich.prompt import FloatPrompt, IntPrompt

    n_particles = IntPrompt.ask("Particle count", default=10_000)
    restitution = FloatPrompt.ask("Collision restitution (normal component, 0..1)", default=0.8)
    v_min_normal = FloatPrompt.ask(
        "Minimum relative normal speed below which collisions are perfectly elastic", default=0.01
    )
    n_steps = IntPrompt.ask("Number of steps", default=5000)
    return RunParams(
        n_particles=n_particles,
        restitution=restitution,
        v_min_normal=v_min_normal,
        n_steps=n_steps,
    )
