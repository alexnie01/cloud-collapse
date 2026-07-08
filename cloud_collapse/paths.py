from __future__ import annotations

import os

__all__ = ["data_path", "output_path"]

DATA_DIR = "data"
OUTPUTS_DIR = "outputs"


def data_path(name: str) -> str:
    """Zarr trajectory path for run `name`: data/<name>/<name>.zarr.

    Pure path computation -- callers that are about to write make the parent
    directory themselves, so a read on a nonexistent run fails with zarr's own
    clear error instead of this function silently creating an empty directory.
    """
    return os.path.join(DATA_DIR, name, f"{name}.zarr")


def output_path(name: str) -> str:
    """Rendered-movie path for run `name`: outputs/<name>/<name>.mp4. See data_path."""
    return os.path.join(OUTPUTS_DIR, name, f"{name}.mp4")
