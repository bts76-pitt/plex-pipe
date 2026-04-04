"""SpatialData zarr helpers (no napari dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import spatialdata as sd

# ---------------------------------------------------------------------------
# Compatibility shim: zarr v3 chunk-shape normalisation
# ---------------------------------------------------------------------------
# ome-zarr 0.14 passes dask block-structure tuples
# (e.g. ``((1, 1), (64, 64), (64, 64))``) as the ``chunks`` storage option
# when writing multiscale images/labels.  zarr ≥ 3.0 requires a flat tuple
# of integers ``(1, 64, 64)``.  This shim patches
# ``spatialdata._io.io_raster.get_pyramid_levels`` so that when it is called
# with ``attr="chunks"`` the per-level block structures are flattened to the
# first (= uniform) block size per axis before they reach zarr.
#
# The patch is idempotent: importing this module multiple times only installs
# the wrapper once.

def _maybe_patch_spatialdata_chunks() -> None:
    try:
        from spatialdata._io import io_raster as _ir
    except ImportError:
        return

    if getattr(_ir, "_plex_pipe_chunk_patch_applied", False):
        return

    _original_gpl = _ir.get_pyramid_levels

    def _patched_get_pyramid_levels(
        image: Any,
        attr: str | None = None,
        n: int | None = None,
    ) -> Any:
        result = _original_gpl(image, attr, n)
        if attr != "chunks" or result is None:
            return result

        def _flatten(spec: Any) -> Any:
            """Convert dask block structure to a flat tuple of ints."""
            if not isinstance(spec, (tuple, list)):
                return spec
            if all(isinstance(c, (tuple, list)) for c in spec):
                # Each element is a per-axis tuple of block sizes → take first.
                return tuple(int(c[0]) for c in spec if len(c) > 0)
            return spec

        if n is not None:
            return _flatten(result)
        return [_flatten(c) for c in result]

    _ir.get_pyramid_levels = _patched_get_pyramid_levels
    _ir._plex_pipe_chunk_patch_applied = True  # type: ignore[attr-defined]


_maybe_patch_spatialdata_chunks()


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
