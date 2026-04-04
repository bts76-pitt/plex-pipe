"""Tests for _normalize_sdata_chunks and _persist_annotation_elements."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import make_minimal_spatialdata_zarr


def _make_mask(h: int, w: int, n_regions: int = 2) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.int32)
    step = h // (n_regions + 1)
    for i in range(1, n_regions + 1):
        r0, r1 = (i - 1) * step, i * step
        mask[r0:r1, : w // 2] = i
    return mask


# ---------------------------------------------------------------------------
# _normalize_sdata_chunks — must cover labels, not just images
# ---------------------------------------------------------------------------

def test_normalize_sdata_chunks_flattens_label_encoding(tmp_path: Path) -> None:
    """Labels2DModel.parse produces block-structure chunks; normalization must flatten them."""
    import spatialdata as sd
    from spatialdata.models import Labels2DModel

    from plex_pipe.ui.region_annotator import _normalize_sdata_chunks

    mask = np.zeros((200, 200), dtype=np.int32)
    mask[10:90, 10:90] = 1

    labels = Labels2DModel.parse(
        data=mask, dims=("y", "x"), scale_factors=[2], chunks=(64, 64)
    )
    sdata = sd.SpatialData(labels={"tissue_regions": labels})

    _normalize_sdata_chunks(sdata)

    for _, node in sdata.labels["tissue_regions"].items():
        for _var_name, var in node.data_vars.items():
            enc = getattr(var, "encoding", {})
            if "chunks" in enc:
                chunks = enc["chunks"]
                assert isinstance(chunks, tuple), f"encoding chunks not a tuple: {chunks}"
                assert all(isinstance(c, int) for c in chunks), (
                    f"encoding chunks contain non-int after normalization: {chunks}"
                )


# ---------------------------------------------------------------------------
# _persist_annotation_elements round-trip
# ---------------------------------------------------------------------------

def test_persist_annotation_elements_v3(tmp_path: Path) -> None:
    """Labels written to a v3 store are readable and the image is preserved."""
    from spatialdata.models import Labels2DModel

    from plex_pipe.ui.region_annotator import _persist_annotation_elements
    from plex_pipe.io.spatialdata_zarr import read_spatialdata_zarr

    src = make_minimal_spatialdata_zarr(tmp_path / "src.zarr", zarr_format=3, with_labels=False)
    sdata = read_spatialdata_zarr(src)
    sdata.labels["tissue_regions"] = Labels2DModel.parse(
        data=_make_mask(128, 128), dims=("y", "x"), scale_factors=[2], chunks=(64, 64)
    )

    dest = tmp_path / "dest.zarr"
    _persist_annotation_elements(sdata, original_path=src, save_path=dest)

    result = read_spatialdata_zarr(dest)
    assert "tissue_regions" in result.labels
    assert "DAPI" in result.images


def test_persist_annotation_elements_v2(tmp_path: Path) -> None:
    """Labels written to a v2 (legacy) store are readable."""
    from spatialdata.models import Labels2DModel

    from plex_pipe.ui.region_annotator import _persist_annotation_elements, _root_zarr_major_version
    from plex_pipe.io.spatialdata_zarr import read_spatialdata_zarr

    src = make_minimal_spatialdata_zarr(tmp_path / "src.zarr", zarr_format=2, with_labels=False)
    if _root_zarr_major_version(src) != 2:
        pytest.skip("zarr v2 store creation not supported in this environment")

    sdata = read_spatialdata_zarr(src)
    sdata.labels["tissue_regions"] = Labels2DModel.parse(
        data=_make_mask(128, 128), dims=("y", "x"), scale_factors=[2], chunks=(64, 64)
    )

    dest = tmp_path / "dest.zarr"
    _persist_annotation_elements(sdata, original_path=src, save_path=dest)

    result = read_spatialdata_zarr(dest)
    assert "tissue_regions" in result.labels
    assert "DAPI" in result.images


# ---------------------------------------------------------------------------
# Fixture smoke tests
# ---------------------------------------------------------------------------

def test_minimal_sdata_zarr_v3_fixture(minimal_sdata_zarr_v3: Path) -> None:
    from plex_pipe.io.spatialdata_zarr import read_spatialdata_zarr
    from plex_pipe.ui.region_annotator import _root_zarr_major_version

    assert _root_zarr_major_version(minimal_sdata_zarr_v3) == 3
    sdata = read_spatialdata_zarr(minimal_sdata_zarr_v3)
    assert "DAPI" in sdata.images
    assert "tissue_regions" in sdata.labels


def test_minimal_sdata_zarr_v2_fixture(minimal_sdata_zarr_v2: Path) -> None:
    from plex_pipe.io.spatialdata_zarr import read_spatialdata_zarr
    from plex_pipe.ui.region_annotator import _root_zarr_major_version

    if _root_zarr_major_version(minimal_sdata_zarr_v2) != 2:
        pytest.skip("zarr v2 store creation not supported in this environment")

    sdata = read_spatialdata_zarr(minimal_sdata_zarr_v2)
    assert "DAPI" in sdata.images
    assert "tissue_regions" in sdata.labels
