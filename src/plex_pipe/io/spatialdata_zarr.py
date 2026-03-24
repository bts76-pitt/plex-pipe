"""SpatialData zarr helpers (no napari dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import spatialdata as sd


def read_spatialdata_zarr(
    path: str | Path,
    *,
    on_bad_elements: Literal["error", "warn"] = "warn",
) -> sd.SpatialData:
    """Load a SpatialData zarr store.

    Parameters
    ----------
    path
        Path to the zarr directory.
    on_bad_elements
        If ``"warn"`` (default), raster subgroups that are incomplete (e.g. interrupted
        writes missing OME-Zarr ``multiscales`` metadata) are skipped with a warning
        instead of aborting the whole read. Use ``"error"`` for strict behavior.

    Notes
    -----
    A failed annotation save can leave ``labels/<name>/`` on disk without valid
    multiscales metadata; strict reads then raise ``KeyError: 'multiscales'``.
    Skipping bad elements lets you open the core and still load intact images;
    delete or replace the broken subgroup if you need a clean store.
    """
    path = str(path)
    if on_bad_elements == "error":
        return sd.read_zarr(path)
    try:
        from spatialdata._io._utils import BadFileHandleMethod
    except ImportError:
        return sd.read_zarr(path)
    return sd.read_zarr(path, on_bad_files=BadFileHandleMethod.WARN)
