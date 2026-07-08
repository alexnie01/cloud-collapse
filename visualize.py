from __future__ import annotations

import time

import cyclopts
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.animation import FuncAnimation
from rich.progress import Progress, track

from cloud_collapse.io.trajectory_store import open_store, read_frame

app = cyclopts.App(help="Stage 2: view a cloud-collapse Zarr trajectory.")

_RED = np.array([255, 0, 0], dtype=np.uint8)
_GREEN = np.array([0, 255, 0], dtype=np.uint8)
_BLACK = np.array([0, 0, 0], dtype=np.uint8)


def status_colors(escaped: np.ndarray, collided: np.ndarray, parked: np.ndarray, kinetic_energy: np.ndarray) -> np.ndarray:
    """Per-particle RGB, straight from stored/derived per-frame data (never recomputed elsewhere):
    green = escaped (still coasting), red = collided sometime since the last recorded
    frame, black = parked (frozen at the origin, invisible against the black background),
    otherwise a white-to-red gradient by kinetic energy (redder = faster), normalized
    against the fastest currently-bound, non-colliding particle in this frame.
    """
    n = len(escaped)
    colors = np.full((n, 3), 255, dtype=np.uint8)

    bound = ~escaped & ~collided & ~parked
    if bound.any():
        ke_bound = kinetic_energy[bound]
        ke_max = ke_bound.max()
        fade = 255 - (255 * ke_bound / ke_max).astype(np.uint8) if ke_max > 0 else np.full(ke_bound.shape, 255, np.uint8)
        colors[bound, 1] = fade
        colors[bound, 2] = fade

    colors[collided & ~escaped & ~parked] = _RED
    colors[escaped & ~parked] = _GREEN
    colors[parked] = _BLACK
    return colors


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

    masses = root["masses"][:]

    def per_particle_ke(velocities: np.ndarray) -> np.ndarray:
        return 0.5 * masses * np.einsum("ij,ij->i", velocities, velocities)

    plotter = pv.Plotter(off_screen=export is not None)
    plotter.set_background("black")
    positions0, velocities0, escaped0, collided0, parked0 = read_frame(root, 0)
    cloud = pv.PolyData(positions0)
    cloud["colors"] = status_colors(escaped0, collided0, parked0, per_particle_ke(velocities0))
    plotter.add_points(cloud, style="points_gaussian", point_size=6, scalars="colors", rgb=True, opacity=0.85)
    plotter.add_mesh(pv.Box(bounds=bounds), style="wireframe", color="gray", opacity=0.4)
    plotter.show_bounds(
        bounds=bounds, grid="back", location="outer", color="white", xtitle="x", ytitle="y", ztitle="z"
    )
    plotter.add_axes(color="white")
    text_actor = plotter.add_text(f"t = {times[0]:.3f}", position="upper_left", font_size=12, color="white")
    plotter.add_text(
        "white->red = kinetic energy   red = collided   green = escaped   black = parked",
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

    arrow_mesh = pv.Arrow(start=(0.0, 0.0, 0.0), direction=L_direction(0), scale=arrow_length)
    plotter.add_mesh(arrow_mesh, color="yellow")
    plotter.add_text("yellow arrow = conserved angular momentum L", position="lower_left", font_size=10, color="yellow")

    plotter.show(auto_close=False, interactive_update=True)
    plotter.reset_camera(bounds=bounds)

    slider_ref = {"widget": None}

    def render_frame(frame_idx: int, paused: bool) -> None:
        positions, velocities, escaped, collided, parked = read_frame(root, frame_idx)
        cloud.points = positions
        cloud["colors"] = status_colors(escaped, collided, parked, per_particle_ke(velocities))
        step_idx = frame_idx * frame_stride
        arrow_mesh.points = pv.Arrow(start=(0.0, 0.0, 0.0), direction=L_direction(step_idx), scale=arrow_length).points
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
    masses = root["masses"][:]

    def per_particle_ke(velocities: np.ndarray) -> np.ndarray:
        return 0.5 * masses * np.einsum("ij,ij->i", velocities, velocities)

    positions0, velocities0, escaped0, collided0, parked0 = read_frame(root, 0)

    # Fixed system size (escape_radius from the run params), matching the PyVista viewer.
    R = float(root.attrs["escape_radius_factor"]) * float(root.attrs["cloud_r_max"])
    arrow_length = 0.5 * R

    def L_direction(step_idx: int) -> np.ndarray:
        L = L_diag[step_idx]
        norm = np.linalg.norm(L)
        return L / norm if norm > 0 else np.array([0.0, 0.0, 1.0])

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    scatter = ax.scatter(
        positions0[:, 0],
        positions0[:, 1],
        positions0[:, 2],
        s=2,
        c=status_colors(escaped0, collided0, parked0, per_particle_ke(velocities0)) / 255.0,
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
        "white->red = kinetic energy   red = collided   green = escaped   black = parked",
        transform=ax.transAxes,
        color="white",
    )
    title = ax.set_title(f"t = {times[0]:.3f}", color="white")
    quiver = [ax.quiver(0, 0, 0, *(arrow_length * L_direction(0)), color="yellow")]
    progress = Progress()
    progress_task = progress.add_task("Playing", total=n_frames)
    progress.start()

    def update(frame_idx: int):
        positions, velocities, escaped, collided, parked = read_frame(root, frame_idx)
        scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        scatter.set_color(status_colors(escaped, collided, parked, per_particle_ke(velocities)) / 255.0)
        title.set_text(f"t = {times[frame_idx]:.3f}")
        quiver[0].remove()
        quiver[0] = ax.quiver(0, 0, 0, *(arrow_length * L_direction(frame_idx * frame_stride)), color="yellow")
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
    one escapes or gets parked); the boiled_* series are plotted separately,
    dotted, just to show how much of each total has drained out of the live
    system over time -- they step up at escape/park events, never down.
    """
    root = open_store(store_path)
    diag = root["diagnostics"]
    frame_stride = int(root.attrs["frame_stride"])
    t_full = diag["step_times"][:]
    ke = diag["kinetic_energy"][:]
    L = diag["angular_momentum"][:]
    boiled_ke = diag["boiled_kinetic_energy"][:]
    boiled_L = diag["boiled_angular_momentum"][:]
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
    axes[1].set_ylabel("Angular momentum")
    axes[1].set_xlabel("Physical time")
    axes[1].legend()

    fig.tight_layout()
    plt.show()


@app.command
def show(store_path: str, fps: int = 30, export: str | None = None) -> None:
    """Animate a trajectory with PyVista (GPU-backed)."""
    animate(store_path, fps=fps, export=export)


@app.command
def fallback(store_path: str) -> None:
    """Animate a trajectory with matplotlib (no VTK/GPU dependency)."""
    matplotlib_fallback(store_path)


@app.command
def diagnostics(store_path: str) -> None:
    """Plot energy and angular momentum conservation diagnostics."""
    plot_diagnostics(store_path)


if __name__ == "__main__":
    app()
