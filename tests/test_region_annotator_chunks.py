import pytest


def test_canonicalize_chunks_tuple_of_ints():
    from plex_pipe.ui.region_annotator import _canonicalize_chunks

    assert _canonicalize_chunks((256, 512), (1000, 2000)) == (256, 512)


def test_canonicalize_chunks_nested_block_structure():
    from plex_pipe.ui.region_annotator import _canonicalize_chunks

    # Mimics dask block layout: per-axis tuple of block sizes (trailing remainder differs).
    nested = (
        (1024, 1024, 208),
        (1024, 1024, 544),
    )
    assert _canonicalize_chunks(nested, (3000, 3000)) == (1024, 1024)


def test_canonicalize_chunks_clamps_to_shape():
    from plex_pipe.ui.region_annotator import _canonicalize_chunks

    assert _canonicalize_chunks((9999, 5), (100, 4)) == (100, 4)


def test_canonicalize_chunks_unparseable_returns_none():
    from plex_pipe.ui.region_annotator import _canonicalize_chunks

    assert _canonicalize_chunks("not-a-chunk-spec", (10, 10)) is None


def test_labels_model_rejects_nested_chunks_without_canonicalization():
    """Safety check: demonstrates why canonicalization exists."""
    from spatialdata.models import Labels2DModel
    import numpy as np

    nested = ((1024, 208), (1024, 544))
    data = np.zeros((32, 32), dtype=np.int32)
    with pytest.raises(Exception):
        Labels2DModel.parse(data=data, dims=("y", "x"), scale_factors=[2, 2], chunks=nested)


def test_labels_model_accepts_canonicalized_chunks():
    from plex_pipe.ui.region_annotator import _canonicalize_chunks
    from spatialdata.models import Labels2DModel
    import numpy as np

    nested = ((1024, 208), (1024, 544))
    data = np.zeros((32, 32), dtype=np.int32)
    chunks = _canonicalize_chunks(nested, data.shape)
    assert chunks == (32, 32)
    Labels2DModel.parse(data=data, dims=("y", "x"), scale_factors=[2, 2], chunks=chunks)

