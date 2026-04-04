import importlib.util
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# =============================================================================
# GLOBAL PATCH (Runs at import time, BEFORE collection)
# =============================================================================

# 1. Check if we are in a headless environment (GitHub Actions)
#    (Or just always mock if you want consistent behavior)
if importlib.util.find_spec("napari") is None:
    # Napari is missing, so we MUST mock it immediately
    print("Headless environment detected: Mocking napari and qtpy...")

    # --- MOCK NAPARI ---
    mock_napari = MagicMock()
    mock_napari.__path__ = []  # Vital: makes it look like a package

    sys.modules["napari"] = mock_napari
    sys.modules["napari.layers"] = MagicMock()
    sys.modules["napari.viewer"] = MagicMock()
    sys.modules["napari.utils"] = MagicMock()
    sys.modules["napari.utils.theme"] = MagicMock()
    sys.modules["napari.resources"] = MagicMock()
    sys.modules["napari.types"] = MagicMock()
    sys.modules["napari._pydantic_compat"] = MagicMock()

    # --- MOCK QTPY ---
    mock_qtpy = MagicMock()
    mock_qtpy.QtWidgets.QFileDialog.getOpenFileName.return_value = ("test.csv", "")
    mock_qtpy.QtWidgets.QFileDialog.getSaveFileName.return_value = ("test.pkl", "")

    sys.modules["qtpy"] = mock_qtpy
    sys.modules["qtpy.QtWidgets"] = mock_qtpy.QtWidgets
    sys.modules["PyQt5"] = MagicMock()  # Prevent backend search


# =============================================================================
# SPATIALDATA ZARR TEST FIXTURES
# =============================================================================

def make_minimal_spatialdata_zarr(
    path: Path,
    *,
    spatial_shape: tuple[int, int] = (512, 512),
    channels: list[str] | None = None,
    chunk_size: tuple[int, int] = (64, 64),
    scale_factors: list[int] | None = None,
    with_labels: bool = True,
    zarr_format: int = 3,
) -> Path:
    """Create a tiny SpatialData Zarr store for use in tests.

    Matches the format produced by ``CoreAssembler``: each channel is a
    separate ``(1, Y, X)`` Image2D element, which is what the annotator
    and quantification stages expect.

    Parameters
    ----------
    path:
        Destination path for the ``.zarr`` directory.
    spatial_shape:
        ``(Y, X)`` spatial dimensions of each channel image.
    channels:
        Channel names.  Defaults to ``["DAPI", "CD3"]``.
    chunk_size:
        ``(Y, X)`` chunk shape for image arrays.
    scale_factors:
        Pyramid downscale factors.  Defaults to ``[2]`` (one extra level).
    with_labels:
        When True, also writes a ``tissue_regions`` Labels2D element.
    zarr_format:
        Zarr store format: ``2`` for legacy v2, ``3`` for current v3.

    Returns
    -------
    Path
        The path passed in (now populated with a valid SpatialData store).
    """
    import contextlib

    import zarr
    import spatialdata as sd
    from spatialdata.models import Image2DModel, Labels2DModel

    # Ensure the zarr v3 chunk-shape compatibility patch is active.
    import plex_pipe.io.spatialdata_zarr  # noqa: F401

    if scale_factors is None:
        scale_factors = [2]
    if channels is None:
        channels = ["DAPI", "CD3"]

    rng = np.random.default_rng(42)
    h, w = spatial_shape
    full_chunk = (1, chunk_size[0], chunk_size[1])

    images: dict[str, Any] = {}
    for ch in channels:
        data = rng.integers(0, 1000, (1, h, w), dtype=np.uint16)
        images[ch] = Image2DModel.parse(
            data=data,
            dims=("c", "y", "x"),
            scale_factors=scale_factors,
            chunks=full_chunk,
        )

    elements: dict[str, Any] = {"images": images}

    if with_labels:
        label_data = rng.integers(1, 5, (h, w), dtype=np.int32)
        label_data[0, 0] = 0  # zarr labels convention: pixel (0,0) = background
        labels = Labels2DModel.parse(
            data=label_data,
            dims=("y", "x"),
            scale_factors=scale_factors,
            chunks=chunk_size,
        )
        elements["labels"] = {"tissue_regions": labels}

    sdata = sd.SpatialData(**elements)

    @contextlib.contextmanager
    def _zarr_format(fmt: int):  # type: ignore[return]
        """Set zarr default format for the duration of the write."""
        try:
            with zarr.config.set({"default_zarr_format": fmt}):
                yield
        except Exception:
            yield  # zarr version doesn't support this config key — use default

    with _zarr_format(zarr_format):
        sdata.write(str(path), overwrite=True)

    return path


@pytest.fixture
def minimal_sdata_zarr_v3(tmp_path: Path) -> Path:
    """Tiny SpatialData Zarr v3 store with one image and one labels element."""
    return make_minimal_spatialdata_zarr(tmp_path / "test_v3.zarr", zarr_format=3)


@pytest.fixture
def minimal_sdata_zarr_v2(tmp_path: Path) -> Path:
    """Tiny SpatialData Zarr v2 (legacy) store with one image and one labels element."""
    return make_minimal_spatialdata_zarr(tmp_path / "test_v2.zarr", zarr_format=2)

