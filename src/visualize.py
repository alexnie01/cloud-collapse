from __future__ import annotations

import time

import cyclopts
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.animation import FuncAnimation
from rich.progress import Progress, track

from cloud_collapse.io.trajectory_store import open_store, read_frame
from cloud_collapse.paths import data_path

app = cyclopts.App(help="Stage 2: view a cloud-collapse Zarr trajectory.")

_RED = np.array([255, 0, 0], dtype=np.uint8)
_GREEN = np.array([0, 255, 0], dtype=np.uint8)
_BLACK = np.array([0, 0, 0], dtype=np.uint8)

# Mass ratio (relative to the original single-particle mass) at which a particle
# switches from a tiny flat "star" glow to a lit, yellow-shaded "planet" sphere.
_PLANET_MASS_RATIO = 3.0
_PLANET_COLOR = np.array([255, 200, 60], dtype=np.uint8)


def status_colors(
    escaped: np.ndarray, collided: np.ndarray, parked: np.ndarray, merged: np.ndarray, kinetic_energy: np.ndarray
) -> np.ndarray:
    """Per-particle RGB, straight from stored/derived per-frame data (never recomputed elsewhere):
    green = escaped (still coasting), red = collided sometime since the last recorded frame,
    black = parked or merged-away (both frozen at the origin, invisible against the black
    background), otherwise a white-to-red gradient by kinetic energy (redder = faster),
    normalized against the fastest currently-bound, non-colliding particle in this frame.
    """
    n = len(escaped)
    colors = np.full((n, 3), 255, dtype=np.uint8)
    frozen = parked | merged

    bound = ~escaped & ~collided & ~frozen
    if bound.any():
        ke_bound = kinetic_energy[bound]
        ke_max = ke_bound.max()
        fade = 255 - (255 * ke_bound / ke_max).astype(np.uint8) if ke_max > 0 else np.full(ke_bound.shape, 255, np.uint8)
        colors[bound, 1] = fade
        colors[bound, 2] = fade

    colors[collided & ~escaped & ~frozen] = _RED
    colors[escaped & ~frozen] = _GREEN
    colors[frozen] = _BLACK
    return colors


def per_particle_radius(masses: np.ndarray, particle_mass: float, base_radius: float) -> np.ndarray:
    """Display radius per particle: base_radius * (mass / particle_mass)**(1/3).

    Constant-density solid-sphere scaling (volume, hence mass, grows as r^3), so a
    merged body visibly grows as it accretes. `particle_mass` is the original
    single-particle mass (mass at t=0, before any merges) -- an unmerged particle
    renders at exactly `base_radius`. Parked/merged-away particles have mass zeroed
    (see integrate.py), so this naturally gives them radius 0 too; harmless since
    they're already rendered invisible (see status_colors).
    """
    return base_radius * (masses / particle_mass) ** (1.0 / 3.0)


def tiered_radii(
    masses: np.ndarray, particle_mass: float, base_radius: float, planet_ratio: float = _PLANET_MASS_RATIO
) -> tuple[np.ndarray, np.ndarray]:
    """Split each particle's display radius into a (star_radius, planet_radius) pair.

    Below `planet_ratio` accreted mass, a particle is a "star": fixed tiny radius, no
    real growth, meant to be rendered flat/emissive (no lighting) like the original
    look before per-particle sizing existed. At/above it, a particle is a "planet":
    real per_particle_radius growth, meant to be rendered as a lit, shaded sphere.
    Each particle gets radius 0 in whichever tier it doesn't currently belong to, so
    both tiers can be drawn as separate fixed-size actors without double-rendering
    any particle.
    """
    is_planet = (masses / particle_mass) >= planet_ratio
    star_radius = np.where(is_planet, 0.0, base_radius)
    planet_radius = np.where(is_planet, per_particle_radius(masses, particle_mass, base_radius), 0.0)
    return star_radius, planet_radius


def animate(store_path: str, fps: int = 30, export: str | None = None) -> None:
    """PyVista GPU-backed 3D animation with lazy per-frame Zarr reads and on-screen physical time.

    export=None opens an interactive window; otherwise renders off-screen straight to a movie
    file (extension picks the codec, e.g. .mp4 or .gif).
    """
    root = open_store(store_path)
    n_frames = root["positions"].shape[0]
    times = root["times"][:]
    frame_stride = int(root.attrs["frame_stride"])
    L_diag = root["diagnostics"]["angular_momentum"]

    # Fixed system size (escape_radius from the run params), not derived from where
    # points actually end up -- an escaped particle coasting outward would otherwise
    # balloon the box each run, shrinking the bound core to a speck.
    R = float(root.attrs["escape_radius_factor"]) * float(root.attrs["cloud_r_max"])
    bounds = (-R, R, -R, R, -R, R)
    arrow_length = 0.5 * R

    # Baseline splat radius (world units) for an unmerged particle -- same formula PyVista
    # itself uses internally to convert point_size -> gaussian scale_factor for a scene of
    # this bounding-box size. point_size is deliberately small so unmerged particles start
    # out as faint pinpricks, leaving accretion (tiered_radii/per_particle_radius) as the
    # dominant visual size cue.
    particle_mass = float(root.attrs["total_mass"]) / float(root.attrs["n_particles"])
    point_size = 1
    base_radius = point_size * np.linalg.norm([2 * R, 2 * R, 2 * R]) / 1300

    def per_particle_ke(velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
        return 0.5 * masses * np.einsum("ij,ij->i", velocities, velocities)

    plotter = pv.Plotter(off_screen=export is not None)
    plotter.set_background("black")

    frame0 = read_frame(root, 0)
    colors0 = status_colors(
        frame0.escaped, frame0.collided, frame0.parked, frame0.merged, per_particle_ke(frame0.velocities, frame0.masses)
    )
    star_radius0, planet_radius0 = tiered_radii(frame0.masses, particle_mass, base_radius)
    planet_colors0 = np.tile(_PLANET_COLOR, (frame0.positions.shape[0], 1))

    # "Stars": tiny, flat, soft-glowing points (render_points_as_spheres=False gives a
    # camera-facing gaussian blob, not a lit 3D sphere) -- matches the plain point-like
    # look particles had before per-particle sizing existed. NB emissive=True makes these
    # invisible with this PyVista/VTK version (verified empirically) -- do not set it.
    star_cloud = pv.PolyData(frame0.positions)
    star_cloud["colors"] = colors0
    star_cloud["radius"] = star_radius0
    star_actor = plotter.add_points(
        star_cloud,
        style="points_gaussian",
        point_size=point_size,
        scalars="colors",
        rgb=True,
        opacity=0.5,
        render_points_as_spheres=False,
    )
    star_actor.mapper.scale_array = "radius"

    # "Planets": real lit spheres (accreted past _PLANET_MASS_RATIO). PointGaussianMapper's
    # sphere shading is a fixed radial-brightness falloff multiplied onto each point's own
    # color -- it does *not* respond to scene lights or Property.specular/ambient (verified
    # empirically: identical render regardless of light/specular color), so "yellow shading"
    # has to come from the point's base color itself, not from lighting. Hence a fixed warm
    # gold color for every planet, replacing the KE-gradient color used for stars.
    planet_cloud = pv.PolyData(frame0.positions)
    planet_cloud["colors"] = planet_colors0
    planet_cloud["radius"] = planet_radius0
    planet_actor = plotter.add_points(
        planet_cloud,
        style="points_gaussian",
        point_size=point_size,
        scalars="colors",
        rgb=True,
        opacity=0.8,
        emissive=False,
        render_points_as_spheres=True,
    )
    planet_actor.mapper.scale_array = "radius"

    plotter.add_mesh(pv.Box(bounds=bounds), style="wireframe", color="gray", opacity=0.4)
    plotter.show_bounds(
        bounds=bounds, grid="back", location="outer", color="white", xtitle="x", ytitle="y", ztitle="z"
    )
    plotter.add_axes(color="white")
    text_actor = plotter.add_text(f"t = {times[0]:.3f}", position="upper_left", font_size=12, color="white")
    plotter.add_text(
        "white->red = kinetic energy   red = collided   green = escaped   black = parked/merged\n"
        f"dot = star   yellow sphere = accreted body (>= {_PLANET_MASS_RATIO:.0f}x mass)",
        position="upper_right",
        font_size=10,
        color="white",
    )

    # Angular momentum vector: read straight from the stored per-step diagnostics
    # (already computed during the run over the whole system, escapees included) --
    # never recomputed here from positions/velocities.
    def L_direction(step_idx: int) -> np.ndarray:
        L = L_diag[step_idx]
        norm = np.linalg.norm(L)
        return L / norm if norm > 0 else np.array([0.0, 0.0, 1.0])

    def make_arrow(direction: np.ndarray) -> pv.PolyData:
        # Narrower than pv.Arrow's defaults (shaft_radius=0.05, tip_radius=0.1) -- this is
        # a reference vector, not something meant to visually compete with the particles.
        return pv.Arrow(
            start=(0.0, 0.0, 0.0), direction=direction, scale=arrow_length, shaft_radius=0.015, tip_radius=0.035
        )

    arrow_mesh = make_arrow(L_direction(0))
    plotter.add_mesh(arrow_mesh, color="yellow", opacity=0.35)
    plotter.add_text("yellow arrow = conserved angular momentum L", position="lower_left", font_size=10, color="yellow")

    plotter.show(auto_close=False, interactive_update=True)
    plotter.reset_camera(bounds=bounds)

    slider_ref = {"widget": None}

    def render_frame(frame_idx: int, paused: bool) -> None:
        frame = read_frame(root, frame_idx)
        colors = status_colors(
            frame.escaped, frame.collided, frame.parked, frame.merged, per_particle_ke(frame.velocities, frame.masses)
        )
        star_radius, planet_radius = tiered_radii(frame.masses, particle_mass, base_radius)
        star_cloud.points = frame.positions
        star_cloud["colors"] = colors
        star_cloud["radius"] = star_radius
        planet_cloud.points = frame.positions
        planet_cloud["radius"] = planet_radius
        step_idx = frame_idx * frame_stride
        arrow_mesh.points = make_arrow(L_direction(step_idx)).points
        label = f"t = {times[frame_idx]:.3f}   frame {frame_idx}/{n_frames - 1}"
        if paused:
            label += "   [PAUSED]"
        text_actor.set_text("upper_left", label)
        if slider_ref["widget"] is not None:
            slider_ref["widget"].GetSliderRepresentation().SetValue(frame_idx)

    if export:
        plotter.open_movie(export, framerate=fps)
        for frame_idx in track(range(n_frames), description="Rendering movie"):
            render_frame(frame_idx, paused=False)
            plotter.render()
            plotter.write_frame()
        plotter.close()
        return

    # Interactive: space pauses/resumes, Left/Right steps one frame at a time (and
    # auto-pauses so you can stop on any frame and freely rotate to check disk
    # thickness before advancing further), or drag the slider to scrub directly.
    plotter.add_text(
        "space = pause/play   ←/→ = step   drag slider = scrub", position="lower_right", font_size=10, color="white"
    )
    state = {"frame_idx": 0, "paused": False}

    def toggle_pause() -> None:
        state["paused"] = not state["paused"]
        render_frame(state["frame_idx"], state["paused"])

    def step(delta: int) -> None:
        state["paused"] = True
        state["frame_idx"] = min(max(state["frame_idx"] + delta, 0), n_frames - 1)
        render_frame(state["frame_idx"], state["paused"])

    def scrub(value: float) -> None:
        state["paused"] = True
        state["frame_idx"] = int(round(min(max(value, 0), n_frames - 1)))
        render_frame(state["frame_idx"], state["paused"])

    plotter.add_key_event("space", toggle_pause)
    plotter.add_key_event("Right", lambda: step(1))
    plotter.add_key_event("Left", lambda: step(-1))
    slider_ref["widget"] = plotter.add_slider_widget(
        scrub,
        rng=[0, n_frames - 1],
        value=0,
        title="frame",
        fmt="%.0f",
        color="white",
        pointa=(0.1, 0.1),
        pointb=(0.9, 0.1),
        interaction_event="always",
    )
    slider_ref["widget"].GetSliderRepresentation().SetLabelHeight(0.015)

    render_frame(0, state["paused"])
    while True:
        try:
            plotter.update()
        except Exception:
            break  # window closed
        if not state["paused"]:
            if state["frame_idx"] >= n_frames - 1:
                state["paused"] = True
                render_frame(state["frame_idx"], state["paused"])
            else:
                state["frame_idx"] += 1
                render_frame(state["frame_idx"], state["paused"])
        else:
            time.sleep(0.05)  # idle while paused, still processing rotation/key events via plotter.update()

    plotter.close()


def matplotlib_fallback(store_path: str) -> None:
    """3D scatter animation via matplotlib, for machines without a working PyVista/VTK GPU path."""
    root = open_store(store_path)
    n_frames = root["positions"].shape[0]
    times = root["times"][:]
    frame_stride = int(root.attrs["frame_stride"])
    L_diag = root["diagnostics"]["angular_momentum"]

    def per_particle_ke(velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
        return 0.5 * masses * np.einsum("ij,ij->i", velocities, velocities)

    frame0 = read_frame(root, 0)

    # Fixed system size (escape_radius from the run params), matching the PyVista viewer.
    R = float(root.attrs["escape_radius_factor"]) * float(root.attrs["cloud_r_max"])
    arrow_length = 0.5 * R

    # matplotlib's scatter `s` is marker *area*, so scale as mass**(2/3) to keep the
    # same constant-density r ~ mass**(1/3) radius growth as the PyVista viewer.
    particle_mass = float(root.attrs["total_mass"]) / float(root.attrs["n_particles"])
    base_size = 0.25

    def marker_sizes(masses: np.ndarray) -> np.ndarray:
        # matplotlib has no real lighting/shading model, so there's no "planet" look to
        # switch to here -- just keep sizes flat (star-like) below the same mass-ratio
        # threshold the PyVista viewer uses, and grow past it, for visual consistency.
        mass_ratio = masses / particle_mass
        is_planet = mass_ratio >= _PLANET_MASS_RATIO
        return np.where(is_planet, base_size * mass_ratio ** (2.0 / 3.0), base_size)

    def L_direction(step_idx: int) -> np.ndarray:
        L = L_diag[step_idx]
        norm = np.linalg.norm(L)
        return L / norm if norm > 0 else np.array([0.0, 0.0, 1.0])

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    scatter = ax.scatter(
        frame0.positions[:, 0],
        frame0.positions[:, 1],
        frame0.positions[:, 2],
        s=marker_sizes(frame0.masses),
        c=status_colors(
            frame0.escaped, frame0.collided, frame0.parked, frame0.merged, per_particle_ke(frame0.velocities, frame0.masses)
        )
        / 255.0,
        alpha=0.6,
    )
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_zlim(-R, R)
    ax.set_xlabel("x", color="white")
    ax.set_ylabel("y", color="white")
    ax.set_zlabel("z", color="white")
    ax.text2D(
        0.0,
        1.02,
        "white->red = kinetic energy   red = collided   green = escaped   black = parked/merged\n"
        "size = accreted mass",
        transform=ax.transAxes,
        color="white",
    )
    title = ax.set_title(f"t = {times[0]:.3f}", color="white")
    quiver = [ax.quiver(0, 0, 0, *(arrow_length * L_direction(0)), color="yellow", alpha=0.35, linewidth=1.0)]
    progress = Progress()
    progress_task = progress.add_task("Playing", total=n_frames)
    progress.start()

    def update(frame_idx: int):
        frame = read_frame(root, frame_idx)
        scatter._offsets3d = (frame.positions[:, 0], frame.positions[:, 1], frame.positions[:, 2])
        scatter.set_color(
            status_colors(
                frame.escaped, frame.collided, frame.parked, frame.merged, per_particle_ke(frame.velocities, frame.masses)
            )
            / 255.0
        )
        scatter.set_sizes(marker_sizes(frame.masses))
        scatter.set_alpha(0.6)  # set_color above resets alpha, so reapply every frame
        title.set_text(f"t = {times[frame_idx]:.3f}")
        quiver[0].remove()
        quiver[0] = ax.quiver(
            0, 0, 0, *(arrow_length * L_direction(frame_idx * frame_stride)), color="yellow", alpha=0.35, linewidth=1.0
        )
        progress.update(progress_task, advance=1)
        if frame_idx == n_frames - 1:
            progress.stop()
        return scatter, title, quiver[0]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / 30, blit=False, repeat=False)
    fig._cloud_collapse_anim = anim  # keep a reference alive
    plt.show()


def plot_diagnostics(store_path: str) -> None:
    """Plot KE, PE, total energy, and angular momentum vs. time.

    KE and angular momentum are recorded every integration step; PE is O(N^2)
    (as expensive as gravity) so it's only recorded at the coarser recorded-
    frame cadence -- hence the two different time axes below. kinetic_energy/
    potential_energy/angular_momentum are already whole-system totals (they
    include the boiled_* contribution moved out of the live particles when
    one escapes or gets parked, and angular_momentum also includes spin banked
    in by merges); the boiled_*/spin series are plotted separately, dotted,
    just to show how much of each total has moved out of the live/orbital
    part over time -- they step up at escape/park/merge events, never down.
    """
    root = open_store(store_path)
    diag = root["diagnostics"]
    frame_stride = int(root.attrs["frame_stride"])
    t_full = diag["step_times"][:]
    ke = diag["kinetic_energy"][:]
    L = diag["angular_momentum"][:]
    boiled_ke = diag["boiled_kinetic_energy"][:]
    boiled_L = diag["boiled_angular_momentum"][:]
    spin_L = diag["spin_angular_momentum"][:]
    t_frames = root["times"][:]
    pe = diag["potential_energy"][:]
    boiled_pe = diag["boiled_potential_energy"][:]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(t_full, ke, label="Kinetic (every step)")
    axes[0].plot(t_frames, pe, label=f"Potential (every {frame_stride} steps)", marker="o", markersize=3)
    ke_at_pe_cadence = ke[::frame_stride]  # same steps write_frame (and thus PE) landed on
    axes[0].plot(t_frames, ke_at_pe_cadence + pe, label="Total (at PE cadence)", linestyle="--")
    axes[0].plot(t_full, boiled_ke, label="Boiled-off kinetic", linestyle=":", color="gray")
    axes[0].plot(t_frames, boiled_pe, label="Boiled-off potential", linestyle=":", color="black")
    axes[0].set_ylabel("Energy")
    axes[0].legend()

    axes[1].plot(t_full, L[:, 0], label="Lx")
    axes[1].plot(t_full, L[:, 1], label="Ly")
    axes[1].plot(t_full, L[:, 2], label="Lz")
    axes[1].plot(t_full, np.linalg.norm(boiled_L, axis=1), label="Boiled-off |L|", linestyle=":", color="gray")
    axes[1].plot(t_full, np.linalg.norm(spin_L, axis=1), label="Spin |L| (merged)", linestyle=":", color="orange")
    axes[1].set_ylabel("Angular momentum")
    axes[1].set_xlabel("Physical time")
    axes[1].legend()

    fig.tight_layout()
    plt.show()


@app.command
def show(name: str, fps: int = 30, export: str | None = None) -> None:
    """Animate a trajectory with PyVista (GPU-backed).

    name: Run name -- reads data/<name>/<name>.zarr (written by simulate.py/run.py).
    export: Movie output path, if given (e.g. outputs/<name>/<name>.mp4); omit for
        an interactive window instead.
    """
    animate(data_path(name), fps=fps, export=export)


@app.command
def fallback(name: str) -> None:
    """Animate a trajectory with matplotlib (no VTK/GPU dependency).

    name: Run name -- reads data/<name>/<name>.zarr (written by simulate.py/run.py).
    """
    matplotlib_fallback(data_path(name))


@app.command
def diagnostics(name: str) -> None:
    """Plot energy and angular momentum conservation diagnostics.

    name: Run name -- reads data/<name>/<name>.zarr (written by simulate.py/run.py).
    """
    plot_diagnostics(data_path(name))


if __name__ == "__main__":
    app()
