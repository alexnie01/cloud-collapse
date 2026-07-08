from __future__ import annotations

import cyclopts
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.animation import FuncAnimation

from cloud_collapse.io.trajectory_store import open_store, read_frame

app = cyclopts.App(help="Stage 2: view a cloud-collapse Zarr trajectory.")


def animate(store_path: str, fps: int = 30, export: str | None = None) -> None:
    """PyVista GPU-backed 3D animation with lazy per-frame Zarr reads and on-screen physical time.

    export=None opens an interactive window; otherwise renders off-screen straight to a movie
    file (extension picks the codec, e.g. .mp4 or .gif).
    """
    root = open_store(store_path)
    n_frames = root["positions"].shape[0]
    times = root["times"][:]

    plotter = pv.Plotter(off_screen=export is not None)
    plotter.set_background("black")
    positions0, _ = read_frame(root, 0)
    cloud = pv.PolyData(positions0)
    plotter.add_points(cloud, style="points_gaussian", point_size=6, color="white", opacity=0.85)
    text_actor = plotter.add_text(f"t = {times[0]:.3f}", position="upper_left", font_size=12, color="white")

    plotter.show(auto_close=False, interactive_update=True)
    if export:
        plotter.open_movie(export, framerate=fps)

    for frame_idx in range(n_frames):
        positions, _ = read_frame(root, frame_idx)
        cloud.points = positions
        text_actor.set_text("upper_left", f"t = {times[frame_idx]:.3f}")
        if export:
            plotter.render()
            plotter.write_frame()
        else:
            plotter.update()

    plotter.close()


def matplotlib_fallback(store_path: str) -> None:
    """3D scatter animation via matplotlib, for machines without a working PyVista/VTK GPU path."""
    root = open_store(store_path)
    n_frames = root["positions"].shape[0]
    times = root["times"][:]
    positions0, _ = read_frame(root, 0)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    scatter = ax.scatter(positions0[:, 0], positions0[:, 1], positions0[:, 2], s=2, c="white")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    limit = np.abs(positions0).max() * 1.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    title = ax.set_title(f"t = {times[0]:.3f}", color="white")

    def update(frame_idx: int):
        positions, _ = read_frame(root, frame_idx)
        scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        title.set_text(f"t = {times[frame_idx]:.3f}")
        return scatter, title

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / 30, blit=False)
    fig._cloud_collapse_anim = anim  # keep a reference alive
    plt.show()


def plot_diagnostics(store_path: str) -> None:
    """Plot KE, PE, total energy, and angular momentum vs. time (full integration-step resolution)."""
    root = open_store(store_path)
    diag = root["diagnostics"]
    t = diag["step_times"][:]
    ke = diag["kinetic_energy"][:]
    pe = diag["potential_energy"][:]
    L = diag["angular_momentum"][:]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(t, ke, label="Kinetic")
    axes[0].plot(t, pe, label="Potential")
    axes[0].plot(t, ke + pe, label="Total", linestyle="--")
    axes[0].set_ylabel("Energy")
    axes[0].legend()

    axes[1].plot(t, L[:, 0], label="Lx")
    axes[1].plot(t, L[:, 1], label="Ly")
    axes[1].plot(t, L[:, 2], label="Lz")
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
