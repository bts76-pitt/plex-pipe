import napari
from magicgui import magicgui
import spatialdata as sd
import numpy as np
import dask.array as da
import pandas as pd
from spatialdata.models import Labels2DModel, TableModel
import anndata as ad
import shutil
from pathlib import Path
from typing import Any

from plex_pipe.io.spatialdata_zarr import read_spatialdata_zarr


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _is_dask_array(x: Any) -> bool:
    return isinstance(x, da.Array)


def _extract_data(x: Any) -> Any:
    """Best-effort extraction of the underlying array payload."""
    if isinstance(x, (np.ndarray, da.Array)):
        return x
    if hasattr(x, "data"):
        try:
            return x.data
        except Exception:
            pass
    return x


def _coarsen_dask_2d(arr: da.Array, factor: int = 2) -> da.Array:
    """Downsample a 2D dask array by `factor` using mean pooling.

    Trims trailing pixels so dimensions are evenly divisible.
    """
    h, w = arr.shape[-2], arr.shape[-1]
    h_trim = h - (h % factor)
    w_trim = w - (w % factor)
    trimmed = arr[..., :h_trim, :w_trim]
    # Reshape into (factor x factor) blocks then take the mean.
    if trimmed.ndim == 2:
        return da.coarsen(np.mean, trimmed, {0: factor, 1: factor}, trim_excess=True)
    # 3D (c, y, x)
    return da.coarsen(np.mean, trimmed, {trimmed.ndim - 2: factor, trimmed.ndim - 1: factor}, trim_excess=True)


def _datatree_to_dask_levels(
    multiscale_element: Any,
    max_chunk: int = 4096,
    gl_max_texture: int = 16384,
) -> list[da.Array]:
    """Convert a SpatialData multiscale element into chunked dask arrays.

    Parameters
    ----------
    multiscale_element
        A DataTree-like object from ``sdata.images[name]``.
    max_chunk : int
        Maximum chunk size per spatial dimension.
    gl_max_texture : int
        ``GL_MAX_TEXTURE_SIZE`` of the GPU.  If the smallest existing
        pyramid level still exceeds this, additional 2x-downsampled levels
        are generated lazily so napari can render without error.
    """

    def scale_idx(key: str) -> int:
        key = str(key)
        try:
            return int(key.replace("scale", ""))
        except ValueError:
            try:
                return int(key)
            except ValueError:
                return 0

    items = sorted(multiscale_element.items(), key=lambda kv: scale_idx(kv[0]))
    levels: list[da.Array] = []
    for _, node in items:
        da_node = next(iter(node.data_vars.values()))
        payload = _extract_data(da_node.data)
        if not isinstance(payload, da.Array):
            payload = da.from_array(np.asarray(payload))

        # Rechunk if any chunk dimension exceeds the limit.
        if any(c > max_chunk for c in payload.chunksize):
            new_chunks = tuple(min(c, max_chunk) for c in payload.chunksize)
            payload = payload.rechunk(new_chunks)

        levels.append(payload)

    # Ensure the coarsest level fits within GL_MAX_TEXTURE_SIZE.
    # If not, keep halving until it does.
    while levels:
        coarsest = levels[-1]
        spatial_shape = coarsest.shape[-2:]  # (y, x)
        if all(s <= gl_max_texture for s in spatial_shape):
            break
        downsampled = _coarsen_dask_2d(coarsest, factor=2)
        # Rechunk the new level too.
        if any(c > max_chunk for c in downsampled.chunksize):
            new_chunks = tuple(min(c, max_chunk) for c in downsampled.chunksize)
            downsampled = downsampled.rechunk(new_chunks)
        levels.append(downsampled)
        # Safety valve: don't generate more than 10 extra levels.
        if len(levels) > 20:
            break

    return levels


def _ensure_background_pixel_zero(mask: Any) -> Any:
    """Ensure pixel (0, 0) is 0 without materializing the full mask."""
    if not _is_dask_array(mask):
        out = np.array(mask, copy=True)
        if out.size:
            out[0, 0] = 0
        return out

    mask = mask.astype(np.int32)

    def _set_pixel_00(block: np.ndarray, block_info: Any | None = None) -> np.ndarray:
        if block_info is None:
            return block
        chunk_location = block_info[None].get("chunk-location") if None in block_info else None
        if chunk_location == (0, 0):
            block = block.copy()
            block[0, 0] = 0
        return block

    return da.map_blocks(
        _set_pixel_00,
        mask,
        dtype=np.int32,
        chunks=mask.chunks,
    )


def _compute_label_counts(mask: Any) -> np.ndarray:
    """Per-label pixel counts (index == label_id, value == count)."""
    mask_i = mask.astype(np.int32) if hasattr(mask, "astype") else np.asarray(mask, dtype=np.int32)
    if _is_dask_array(mask_i):
        counts = da.bincount(mask_i.ravel()).compute()
        return np.asarray(counts, dtype=np.int64)
    mask_np = np.asarray(mask_i)
    return np.asarray(np.bincount(mask_np.ravel()), dtype=np.int64)


def _compute_max_label(mask: Any) -> int:
    if _is_dask_array(mask):
        return int(da.max(mask).compute())
    return int(np.max(mask))


def _canonicalize_chunks(chunks: Any, shape: tuple[int, ...]) -> tuple[int, ...] | None:
    """Convert various chunk specs to a plain tuple of ints.

    SpatialData elements loaded from disk can expose ``.chunks`` as a Dask block
    structure (tuple-of-tuples). Zarr v3 requires ``chunk_shape`` to be an
    iterable of integers, so we normalize to a simple tuple[int, ...].
    """
    if chunks is None:
        return None
    try:
        # If it's already a tuple of ints, keep it.
        if isinstance(chunks, tuple) and all(isinstance(c, int) for c in chunks):
            out = chunks
        elif isinstance(chunks, (list, tuple)):
            out_list: list[int] = []
            for c, sh in zip(chunks, shape, strict=False):
                if isinstance(c, int):
                    out_list.append(c)
                elif isinstance(c, (list, tuple)) and len(c) > 0 and isinstance(c[0], int):
                    # Take first block size for that axis.
                    out_list.append(int(c[0]))
                else:
                    return None
            out = tuple(out_list)
        else:
            return None
        # Clamp to valid sizes.
        return tuple(int(max(1, min(c, sh))) for c, sh in zip(out, shape, strict=False))
    except Exception:
        return None


def _zarr_store_copy_ignore(_src: str, names: list[str]) -> set[str]:
    """Skip macOS junk that breaks Zarr hierarchy readers."""
    return {n for n in names if n == ".DS_Store" or n.startswith("._")}


# Used by CLI scripts that ``copytree`` zarr stores before partial writes.
zarr_store_copy_ignore = _zarr_store_copy_ignore


def _root_zarr_major_version(store_path: Path) -> int | None:
    """Return 2 or 3 for the on-disk Zarr *root* layout, or None if unknown.

    SpatialData stores are either legacy Zarr v2 (``.zgroup``) or Zarr v3
    (``zarr.json`` at the store root). Writers must use matching OME-NGFF /
    SpatialData raster formats: v2 roots need ``RasterFormatV02`` (NGFF 0.4),
    v3 roots use the current default (NGFF 0.5).
    """
    store_path = Path(store_path)
    if (store_path / "zarr.json").exists():
        return 3
    if (store_path / ".zgroup").exists():
        return 2
    return None


def _write_element_kwargs_for_zarr_root(store_path: Path) -> dict[str, Any]:
    """Keyword args for ``SpatialData.write_element`` matching store Zarr version."""
    major = _root_zarr_major_version(store_path)
    if major != 2:
        return {}

    try:
        from spatialdata._io.format import (
            RasterFormatV02,
            SpatialDataContainerFormatV01,
            TablesFormatV01,
        )
    except ImportError:
        return {}

    # Zarr v2 root: must not use default NGFF 0.5 / RasterFormatV03 writer.
    return {
        "sdata_formats": [
            SpatialDataContainerFormatV01(),
            RasterFormatV02(),
            TablesFormatV01(),
        ],
    }


def _persist_annotation_elements(
    sdata_obj: sd.SpatialData,
    original_path: Path,
    save_path: Path,
) -> None:
    """Persist annotation elements without rewriting image pyramids.

    Writing full ``SpatialData`` can fail on some zarr-v3 stacks when image
    chunk metadata is inherited from existing stores. This routine copies the
    source store (for new output paths) and writes only labels/tables changed
    by this annotator.
    """
    original_path = Path(original_path)
    save_path = Path(save_path)

    if save_path.resolve() != original_path.resolve():
        shutil.copytree(
            original_path,
            save_path,
            dirs_exist_ok=True,
            ignore=_zarr_store_copy_ignore,
        )

    write_kw = _write_element_kwargs_for_zarr_root(save_path)

    target = read_spatialdata_zarr(save_path)

    if "tissue_regions" in sdata_obj.labels:
        target.labels["tissue_regions"] = sdata_obj.labels["tissue_regions"]
        target.write_element("tissue_regions", overwrite=True, **write_kw)

    if "tissue_regions_table" in sdata_obj.tables:
        target.tables["tissue_regions_table"] = sdata_obj.tables["tissue_regions_table"]
        target.write_element("tissue_regions_table", overwrite=True, **write_kw)


def sync_spatialdata_to_store(
    sdata_obj: sd.SpatialData,
    store_path: str | Path,
    *,
    labels: tuple[str, ...] = (),
    tables: tuple[str, ...] = (),
) -> None:
    """Write selected labels/tables into an on-disk SpatialData zarr (no image rewrite).

    Chooses NGFF / SpatialData writers compatible with Zarr v2 vs v3 *store roots*.
    The store must already exist (e.g. after ``shutil.copytree`` from a template).
    """
    store_path = Path(store_path)
    write_kw = _write_element_kwargs_for_zarr_root(store_path)
    target = read_spatialdata_zarr(store_path)
    for name in labels:
        if name not in sdata_obj.labels:
            raise KeyError(f"Label {name!r} not in in-memory SpatialData")
        target.labels[name] = sdata_obj.labels[name]
        target.write_element(name, overwrite=True, **write_kw)
    for name in tables:
        if name not in sdata_obj.tables:
            raise KeyError(f"Table {name!r} not in in-memory SpatialData")
        target.tables[name] = sdata_obj.tables[name]
        target.write_element(name, overwrite=True, **write_kw)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class RegionAnnotationWidget:
    # Downsampling factor for the painting mask.  The mask is painted at
    # ``full_res / downsample`` and scaled back up on save.  For coarse
    # tissue-level annotation (ducts, lobules, stroma) this is plenty.
    _DEFAULT_DOWNSAMPLE: int = 8

    def __init__(
        self,
        viewer: napari.Viewer,
        sdata: sd.SpatialData,
        image_name: str,
        img_shape: tuple[int, int],
        downsample: int | None = None,
    ):
        self.viewer = viewer
        self.sdata = sdata
        self.image_name = image_name
        self.annotations: dict[int, dict] = {}

        # Decide on downsampling.
        self.full_shape = img_shape
        self.downsample = downsample or self._DEFAULT_DOWNSAMPLE
        self.paint_shape = (
            img_shape[0] // self.downsample,
            img_shape[1] // self.downsample,
        )
        # Scale transform so the labels layer aligns with the full-res images.
        self.label_scale = (self.downsample, self.downsample)

        existing_regions = "tissue_regions" in sdata.labels

        if existing_regions:
            print("Loading existing tissue regions...")
            existing_mask = sd.get_pyramid_levels(
                sdata.labels["tissue_regions"], n=0
            ).squeeze()
            existing_mask = _extract_data(existing_mask)

            self.current_region_id = (
                int(da.max(existing_mask).compute())
                if _is_dask_array(existing_mask)
                else int(np.max(existing_mask))
            )

            if "tissue_regions_table" in sdata.tables:
                region_table = sdata.tables["tissue_regions_table"]
                for _, row in region_table.obs.iterrows():
                    region_id = int(row["region_id"])
                    self.annotations[region_id] = {
                        "region_id": region_id,
                        "tissue_type": row.get("tissue_type", "Unknown"),
                        "phenotype": row.get("phenotype", ""),
                        "additional_notes": row.get("notes", ""),
                    }
                print(f"Loaded {len(self.annotations)} existing annotations")

            # Downsample existing mask for painting.
            print(f"Downsampling existing mask {existing_mask.shape} -> {self.paint_shape} ({self.downsample}x)...")
            if _is_dask_array(existing_mask):
                existing_mask = existing_mask.compute()
            existing_mask = np.asarray(existing_mask)
            paint_mask = existing_mask[:: self.downsample, :: self.downsample].copy()
            # Ensure shape matches exactly (rounding).
            paint_mask = paint_mask[: self.paint_shape[0], : self.paint_shape[1]]

            self.regions_layer = self.viewer.add_labels(
                paint_mask,
                name="tissue_regions",
                scale=self.label_scale,
                translate=(0, 0),
            )
            # Force the brush to work in displayed coordinates.
            self.regions_layer.brush_size = max(1, 10 // self.downsample)
        else:
            print(f"Starting fresh annotations (paint resolution: {self.paint_shape}, {self.downsample}x downsampled)...")
            self.current_region_id = 1
            self.regions_layer = self.viewer.add_labels(
                np.zeros(self.paint_shape, dtype=np.int32),
                name="tissue_regions",
                scale=self.label_scale,
                translate=(0, 0),
            )
            self.regions_layer.brush_size = max(1, 10 // self.downsample)

        # Optionally load cell segmentation as a read-only reference layer.
        if "instanseg_cell" in sdata.labels:
            try:
                cell_data = sd.get_pyramid_levels(
                    sdata.labels["instanseg_cell"], n=0
                ).squeeze()
                cell_data = _extract_data(cell_data)
                cell_layer = self.viewer.add_labels(
                    cell_data,          # dask is fine here (read-only)
                    name="cells",
                    opacity=0.3,
                )
                cell_layer.editable = False
            except Exception as e:
                print(f"Could not load cell segmentation: {e}")

        self.regions_layer.mode = "paint"
        self.regions_layer.selected_label = self.current_region_id

        self._create_widget()

    # -- helpers -------------------------------------------------------------

    def _compute_max_mask_label(self) -> int:
        return _compute_max_label(self.regions_layer.data)

    def _compute_present_mask_labels(self) -> np.ndarray:
        counts = _compute_label_counts(self.regions_layer.data)
        present = np.nonzero(counts)[0]
        present = present[present != 0]
        return np.sort(present.astype(int))

    def _region_exists_in_mask(self, region_id: int) -> bool:
        mask_data = self.regions_layer.data
        if _is_dask_array(mask_data):
            return bool(da.any(mask_data == region_id).compute())
        return bool(np.any(mask_data == region_id))

    def _compute_mask_label_counts(self) -> np.ndarray:
        return _compute_label_counts(self.regions_layer.data)

    # -- magicgui widgets ----------------------------------------------------

    def _create_widget(self):
        @magicgui(
            call_button="Annotate Current Region",
            tissue_type={"choices": ["Duct", "Lobule", "Stroma", "Other"]},
            phenotype={"label": "Phenotype Notes"},
            additional_notes={"label": "Additional Notes", "widget_type": "TextEdit"},
        )
        def annotation_widget(
            tissue_type: str = "Duct",
            phenotype: str = "",
            additional_notes: str = "",
        ):
            region_id = self.regions_layer.selected_label
            if region_id == 0:
                print("Please select a non-zero label!")
                return

            action = "Updating" if region_id in self.annotations else "Creating new"
            print(f"{action} annotation for region {region_id}")

            self.annotations[region_id] = {
                "region_id": region_id,
                "tissue_type": tissue_type,
                "phenotype": phenotype,
                "additional_notes": additional_notes,
            }
            print(f"Annotated region {region_id}: {tissue_type} - {phenotype}")

        @magicgui(call_button="New Region (Next ID)")
        def new_region_widget():
            max_mask_label = self._compute_max_mask_label()
            max_ann_label = max(self.annotations.keys()) if self.annotations else 0
            max_existing = max(max_mask_label, max_ann_label)
            self.current_region_id = 1 if max_existing <= 0 else max_existing + 1

            self.regions_layer.selected_label = self.current_region_id
            print(f"Started new region: {self.current_region_id}")
            print("Paint your region, then click 'Annotate Current Region'")

        @magicgui(
            call_button="Select Region to Work On",
            region_id={"label": "Region ID"},
        )
        def select_region_widget(region_id: int = 1):
            self.regions_layer.selected_label = region_id
            self.current_region_id = region_id

            if region_id in self.annotations:
                ann = self.annotations[region_id]
                print(f"\nSelected existing region {region_id}:")
                print(f"  Tissue type: {ann['tissue_type']}")
                print(f"  Phenotype: {ann['phenotype']}")
                print(f"  Notes: {ann['additional_notes']}")
                print("\nYou can now:")
                print("  - Paint to add/modify the mask")
                print("  - Use eraser to remove parts")
                print("  - Click 'Annotate Current Region' to update metadata")
            else:
                print(f"\nSelected region {region_id} (new region)")
                print("Paint your region, then click 'Annotate Current Region'")

        @magicgui(call_button="List All Regions")
        def list_regions_widget():
            mask_ids = set(self._compute_present_mask_labels().tolist())
            annotated_ids = set(self.annotations.keys())
            all_ids = mask_ids.union(annotated_ids)

            if not all_ids:
                print("No regions exist yet")
                return

            print("\n=== Existing Regions ===")
            for region_id in sorted(all_ids):
                parts = []
                if region_id in mask_ids:
                    parts.append("has mask")
                if region_id in annotated_ids:
                    parts.append(self.annotations[region_id]["tissue_type"])
                status = ", ".join(parts) if parts else "no data"
                marker = " <- CURRENT" if region_id == self.regions_layer.selected_label else ""
                print(f"  Region {region_id}: {status}{marker}")

        @magicgui(
            call_button="Delete Region",
            region_id={"label": "Region ID to delete"},
        )
        def delete_region_widget(region_id: int = 1):
            mask_data = self.regions_layer.data
            if self._region_exists_in_mask(region_id):
                if _is_dask_array(mask_data):
                    try:
                        self.regions_layer.data = da.where(
                            mask_data == region_id, 0, mask_data
                        )
                    except Exception:
                        mask_np = np.asarray(mask_data)
                        mask_np[mask_np == region_id] = 0
                        self.regions_layer.data = mask_np
                else:
                    mask_np = np.array(mask_data, copy=True)
                    mask_np[mask_np == region_id] = 0
                    self.regions_layer.data = mask_np
                print(f"Removed region {region_id} from mask")

            if region_id in self.annotations:
                del self.annotations[region_id]
                print(f"Deleted annotation for region {region_id}")

            print(f"Region {region_id} fully deleted")

        @magicgui(call_button="Save Regions to SpatialData")
        def save_widget():
            if not self.annotations:
                print("No annotations to save!")
                return

            # Save at paint resolution. No upscaling needed.
            # The downsample factor is stored so downstream code can map
            # full-res coordinates to label coordinates.
            mask_data = _ensure_background_pixel_zero(self.regions_layer.data.copy())
            label_counts = _compute_label_counts(mask_data)
            unique_labels = np.nonzero(label_counts)[0]
            unique_labels = unique_labels[unique_labels != 0]

            if unique_labels.size == 0:
                print("No regions drawn!")
                return

            unannotated = [int(lid) for lid in unique_labels if int(lid) not in self.annotations]
            if unannotated:
                print(f"Warning: Regions {unannotated} have masks but no annotations")

            # Preserve transformations / chunk info from existing labels.
            transformations = None
            existing_chunks = None
            existing_scale_factors = None

            if "tissue_regions" in self.sdata.labels:
                existing_labels = self.sdata.labels["tissue_regions"]
                try:
                    if hasattr(existing_labels, "transform"):
                        transformations = existing_labels.transform
                    if hasattr(existing_labels, "chunks"):
                        existing_chunks = existing_labels.chunks
                    existing_scale_factors = [2, 2]
                except Exception as e:
                    print(f"Note: Could not read attributes from existing labels: {e}")

            if transformations is None:
                try:
                    ref_element = self.sdata.images[self.image_name]
                    if hasattr(ref_element, "transform"):
                        transformations = ref_element.transform
                except Exception as e:
                    print(f"Note: Could not copy transformations from reference image: {e}")

            ref_shape = mask_data.shape
            chunks = _canonicalize_chunks(existing_chunks, ref_shape) or (
                min(1024, ref_shape[0]),
                min(1024, ref_shape[1]),
            )
            scale_factors = existing_scale_factors or [2, 2]

            labels_model = Labels2DModel.parse(
                data=mask_data.astype(np.int32),
                dims=("y", "x"),
                scale_factors=scale_factors,
                chunks=chunks,
            )
            if transformations is not None:
                labels_model.transform = transformations

            self.sdata.labels["tissue_regions"] = labels_model

            # Build annotation table.
            existing_table = self.sdata.tables.get("tissue_regions_table")
            if existing_table is not None:
                print("Found existing table, merging annotations...")

            obs_data = [
                {
                    "region_id": 0,
                    "area": 0,
                    "tissue_type": "Unknown",
                    "phenotype": "",
                    "notes": "Background/Unpainted",
                }
            ]

            for label_id in sorted(unique_labels):
                lid = int(label_id)
                area = int(label_counts[lid])

                existing_row = None
                if existing_table is not None:
                    rows = existing_table.obs[existing_table.obs["region_id"] == lid]
                    if len(rows) > 0:
                        existing_row = rows.iloc[0]

                if lid in self.annotations:
                    ann = self.annotations[lid]
                elif existing_row is not None:
                    ann = {
                        "tissue_type": existing_row.get("tissue_type", "Unknown"),
                        "phenotype": existing_row.get("phenotype", ""),
                        "additional_notes": existing_row.get("notes", ""),
                    }
                else:
                    ann = {"tissue_type": "Unknown", "phenotype": "", "additional_notes": ""}

                obs_data.append(
                    {
                        "region_id": lid,
                        "area": area,
                        "tissue_type": ann["tissue_type"],
                        "phenotype": ann["phenotype"],
                        "notes": ann["additional_notes"],
                    }
                )

            obs_df = pd.DataFrame(obs_data).reset_index(drop=True)
            obs_df["tissue_type"] = pd.Categorical(obs_df["tissue_type"])
            obs_df["phenotype"] = pd.Categorical(obs_df["phenotype"])
            obs_df["region"] = "tissue_regions"

            uns_metadata = {}
            if existing_table is not None:
                existing_adata = getattr(existing_table, "adata", None)
                if existing_adata is not None and hasattr(existing_adata, "uns"):
                    uns_metadata.update(
                        {k: v for k, v in existing_adata.uns.items() if k != "spatialdata_attrs"}
                    )

            adata = ad.AnnData(
                X=np.zeros((len(obs_df), 0)),
                obs=obs_df,
                uns=uns_metadata,
            )
            table_model = TableModel.parse(
                adata,
                region="tissue_regions",
                region_key="region",
                instance_key="region_id",
            )
            self.sdata.tables["tissue_regions_table"] = table_model

            print(f"Saved {len(unique_labels)} regions to sdata object (in memory)")
            print("Preserved existing attributes and transformations")
            print("Close napari and choose save option to write to disk")

        @magicgui(call_button="Show Summary")
        def summary_widget():
            if not self.annotations:
                print("No annotations yet!")
                return

            print("\n=== Region Annotations Summary ===")
            tissue_counts: dict[str, int] = {}
            for ann in self.annotations.values():
                t = ann["tissue_type"]
                tissue_counts[t] = tissue_counts.get(t, 0) + 1

            print("\nCounts by tissue type:")
            for tissue, count in tissue_counts.items():
                print(f"  {tissue}: {count}")

            print("\nDetailed annotations:")
            for region_id in sorted(self.annotations.keys()):
                ann = self.annotations[region_id]
                print(f"\nRegion {region_id}:")
                print(f"  Type: {ann['tissue_type']}")
                print(f"  Phenotype: {ann['phenotype']}")
                print(f"  Notes: {ann['additional_notes']}")

        # Dock widgets
        self.viewer.window.add_dock_widget(new_region_widget, name="New Region", area="right")
        self.viewer.window.add_dock_widget(select_region_widget, name="Select Region", area="right")
        self.viewer.window.add_dock_widget(list_regions_widget, name="List Regions", area="right")
        self.viewer.window.add_dock_widget(annotation_widget, name="Annotate", area="right")
        self.viewer.window.add_dock_widget(delete_region_widget, name="Delete Region", area="right")
        self.viewer.window.add_dock_widget(save_widget, name="Save", area="right")
        self.viewer.window.add_dock_widget(summary_widget, name="Summary", area="right")


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------

def _is_jupyter() -> bool:
    """Return True when running inside a Jupyter/IPython kernel."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


def launch_region_annotation(sdata_path, auto_backup=True, channels=None, downsample=None):
    """Launch napari for region annotation with all channels.

    In a **Jupyter notebook** the viewer opens non-blocking and the function
    returns ``(sdata, widget)`` immediately so you can keep using the
    notebook while annotating.  Call ``save_annotations(sdata, sdata_path)``
    when you're done.

    From a **standalone script** the viewer blocks until closed, then an
    interactive save dialog is presented in the terminal.

    Parameters
    ----------
    sdata_path : str or Path
        Path to the SpatialData object.
    auto_backup : bool
        If True, suggests saving to a new path with '_annotated' suffix.

    Returns
    -------
    sdata : SpatialData
        The (potentially modified) SpatialData object.
    widget : RegionAnnotationWidget
        The annotation widget (useful for programmatic access in notebooks).
    """
    sdata = read_spatialdata_zarr(sdata_path)
    if not sdata.images:
        raise ValueError(
            f"No readable image layers in {sdata_path!s}. The zarr store may be badly corrupted "
            "(e.g. interrupted write). Try opening the original core, or remove broken subgroups under "
            "`images/` or `labels/` that lack OME-Zarr multiscales metadata."
        )
    viewer = napari.Viewer()

    colormaps = {
        "DAPI": "blue",
        "CK818": "green",
        "CK14": "cyan",
        "ECad": "red",
        "ER": "yellow",
        "HER2": "magenta",
        "GATA3": "green",
        "FOXA1": "cyan",
        "AR": "red",
        "AP2A": "yellow",
        "AP2B": "magenta",
        "CD45": "red",
    }

    # Reference shape (no materialization).
    ref_image_name = list(sdata.images.keys())[0]
    ref_levels = _datatree_to_dask_levels(sdata.images[ref_image_name])
    ref_img_data = ref_levels[0].squeeze()
    if ref_img_data.ndim == 3 and ref_img_data.shape[0] == 1:
        ref_img_data = ref_img_data[0]
    img_shape = ref_img_data.shape  # (y, x)

    # Load image channels lazily as multiscale pyramids.
    # If channels is provided, only load those (saves memory and startup time).
    available = list(sdata.images.keys())
    if channels is not None:
        load_channels = [c for c in channels if c in available]
        skipped = [c for c in available if c not in channels]
        if skipped:
            print(f"Skipping {len(skipped)} channels: {skipped}")
    else:
        load_channels = available

    print(f"Loading {len(load_channels)} channels:")
    for img_name in load_channels:
        print(f"  - {img_name}")
        levels = _datatree_to_dask_levels(sdata.images[img_name])
        levels = [lvl.squeeze() for lvl in levels]
        levels = [
            (lvl[0] if lvl.ndim == 3 and lvl.shape[0] == 1 else lvl) for lvl in levels
        ]

        viewer.add_image(
            levels,
            name=img_name,
            colormap=colormaps.get(img_name, "gray"),
            blending="additive",
            visible=(img_name == "DAPI"),
            multiscale=True,
        )

    widget = RegionAnnotationWidget(viewer, sdata, ref_image_name, img_shape)

    print("\n=== Napari Region Annotation Tool ===")
    print("\nWorkflow:")
    print("1. Click 'New Region' to start with next available ID")
    print("2. OR click 'Select Region' to choose a specific ID to work on")
    print("3. Paint your region with the brush tool")
    print("4. Click 'Annotate Current Region' and fill in metadata")
    print("5. Click 'List Regions' to see all existing regions")
    print("6. Repeat for all regions")
    print("7. Click 'Save' to save to sdata (in memory)")
    print("8. Close napari to write to disk")
    print("\nTips:")
    print("- You can toggle channel visibility to see tissue structure")
    print("- Use eraser tool to fix mistakes")
    print("- The current region ID is shown in napari's label controls")

    in_jupyter = _is_jupyter()

    if in_jupyter:
        # In Jupyter the Qt event loop is already running (via %gui qt5).
        # Blocking would deadlock or hide the window.
        viewer.show(block=False)
        print("\n--- Notebook mode ---")
        print("The napari window should now be visible.")
        print("When you're done annotating, run in a new cell:")
        print(f"    save_annotations(sdata, '{sdata_path}')")
        return sdata, widget

    # Standalone script: block until the viewer is closed.
    viewer.show(block=True)
    print("\n=== Napari closed ===")

    original_path = Path(sdata_path)
    if auto_backup:
        suggested_path = original_path.parent / f"{original_path.stem}_annotated.zarr"
    else:
        suggested_path = original_path

    print(f"\nOriginal path: {original_path}")
    print(f"Suggested save path: {suggested_path}")

    save_choice = input(
        "\nSave options:\n"
        "  1. Save to suggested path (safe)\n"
        "  2. Overwrite original\n"
        "  3. Specify custom path\n"
        "  4. Don't save\n"
        "Choice (1-4): "
    )

    def _normalize_sdata_chunks(sdata_obj):
        """Normalize image chunk metadata and chunking for zarr-v3 writes.

        Some inputs carry xarray encodings where ``encoding["chunks"]`` is a
        Dask block structure (tuple-of-tuples) instead of a chunk-shape tuple.
        zarr v3 rejects this with ``Expected an iterable of integers``.
        """
        for img_name in list(sdata_obj.images.keys()):
            element = sdata_obj.images[img_name]
            # Walk the DataTree and rechunk each variable.
            for _, node in element.items():
                for var_name, var in list(node.data_vars.items()):
                    data = var.data
                    new_var = var
                    if _is_dask_array(data):
                        # Convert to canonical chunk-shape tuple of integers.
                        # Keep chunks <= axis length and avoid irregular grids.
                        uniform = tuple(
                            int(max(1, min(cs, sh)))
                            for cs, sh in zip(data.chunksize, data.shape)
                        )
                        data = data.rechunk(uniform)
                        new_var = var.copy(data=data)
                        # Remove stale/invalid chunk metadata and write a
                        # canonical integer chunk shape expected by zarr v3.
                        if hasattr(new_var, "encoding"):
                            old_enc = dict(getattr(new_var, "encoding", {}))
                            old_enc.pop("chunks", None)
                            old_enc.pop("preferred_chunks", None)
                            old_enc["chunks"] = uniform
                            new_var.encoding = old_enc
                    elif hasattr(new_var, "encoding"):
                        # Non-dask payloads can still carry invalid chunks from
                        # source metadata; drop them and let writer choose.
                        old_enc = dict(getattr(new_var, "encoding", {}))
                        old_enc.pop("chunks", None)
                        old_enc.pop("preferred_chunks", None)
                        new_var.encoding = old_enc
                    node[var_name] = new_var

    save_path = None
    if save_choice == "1":
        save_path = suggested_path
    elif save_choice == "2":
        confirm = input(f"Really overwrite {original_path}? Type 'yes' to confirm: ")
        if confirm == "yes":
            save_path = original_path
        else:
            print("Save cancelled")
    elif save_choice == "3":
        save_path = Path(input("Enter path: "))
        if not save_path.is_absolute():
            save_path = original_path.parent / save_path

    if save_path is not None:
        print("Saving annotations without rewriting image pyramids...")
        _normalize_sdata_chunks(sdata)
        _persist_annotation_elements(
            sdata_obj=sdata,
            original_path=original_path,
            save_path=Path(save_path),
        )
        print(f"Saved to {save_path}")
    else:
        print("Changes not saved to disk")

    return sdata, widget


def save_annotations(sdata, sdata_path, auto_backup=True):
    """Save annotations to disk (call from a notebook after annotating).

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object returned by ``launch_region_annotation``.
    sdata_path : str or Path
        Original path used to load the data.
    auto_backup : bool
        If True, saves to ``<name>_annotated.zarr`` instead of overwriting.
    """
    original_path = Path(sdata_path)
    if auto_backup:
        save_path = original_path.parent / f"{original_path.stem}_annotated.zarr"
    else:
        save_path = original_path

    _persist_annotation_elements(
        sdata_obj=sdata,
        original_path=original_path,
        save_path=save_path,
    )
    print(f"Saved to {save_path}")


# ---------------------------------------------------------------------------
# Standalone export / linking utilities
# ---------------------------------------------------------------------------

def export_regions_to_spatialdata(
    sdata,
    mask_data,
    annotations,
    image_name,
    region_labels_name="tissue_regions",
    table_name="tissue_regions_table",
):
    """Export drawn regions and annotations to a SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object to update.
    mask_data : array-like
        The region mask (2D integer array), may be numpy or dask.
    annotations : dict
        ``{region_id: {'tissue_type': ..., 'phenotype': ..., 'additional_notes': ...}}``.
    image_name : str
        Reference image name for coordinate system.
    region_labels_name : str
        Name for the labels layer.
    table_name : str
        Name for the annotations table.

    Returns
    -------
    SpatialData
        Updated SpatialData object.
    """
    mask_data = _ensure_background_pixel_zero(mask_data)
    label_counts = _compute_label_counts(mask_data)
    unique_labels = np.nonzero(label_counts)[0]
    unique_labels = unique_labels[unique_labels != 0]

    if unique_labels.size == 0:
        print("No regions to export!")
        return sdata

    # Preserve existing metadata.
    transformations = None
    existing_chunks = None
    existing_scale_factors = None

    if region_labels_name in sdata.labels:
        existing_labels = sdata.labels[region_labels_name]
        try:
            if hasattr(existing_labels, "transform"):
                transformations = existing_labels.transform
            if hasattr(existing_labels, "chunks"):
                existing_chunks = existing_labels.chunks
            existing_scale_factors = [2, 2]
        except Exception as e:
            print(f"Note: Could not read attributes from existing labels: {e}")

    if transformations is None:
        try:
            ref_element = sdata.images[image_name]
            if hasattr(ref_element, "transform"):
                transformations = ref_element.transform
        except Exception:
            pass

    ref_shape = mask_data.shape
    chunks = _canonicalize_chunks(existing_chunks, ref_shape) or (
        min(1024, ref_shape[0]),
        min(1024, ref_shape[1]),
    )
    scale_factors = existing_scale_factors or [2, 2]

    labels_model = Labels2DModel.parse(
        data=mask_data.astype(np.int32),
        dims=("y", "x"),
        scale_factors=scale_factors,
        chunks=chunks,
    )
    if transformations is not None:
        labels_model.transform = transformations

    sdata.labels[region_labels_name] = labels_model

    existing_table = sdata.tables.get(table_name)

    obs_data = [
        {
            "region_id": 0,
            "area": 1,
            "tissue_type": "Background",
            "phenotype": "",
            "notes": "Background region at pixel (0,0)",
        }
    ]

    for label_id in sorted(unique_labels):
        lid = int(label_id)
        area = int(label_counts[lid])

        existing_row = None
        if existing_table is not None:
            rows = existing_table.obs[existing_table.obs["region_id"] == lid]
            if len(rows) > 0:
                existing_row = rows.iloc[0]

        if lid in annotations:
            ann = annotations[lid]
        elif existing_row is not None:
            ann = {
                "tissue_type": existing_row.get("tissue_type", "Unknown"),
                "phenotype": existing_row.get("phenotype", ""),
                "additional_notes": existing_row.get("notes", ""),
            }
        else:
            ann = {"tissue_type": "Unknown", "phenotype": "", "additional_notes": ""}

        obs_data.append(
            {
                "region_id": lid,
                "area": area,
                "tissue_type": ann["tissue_type"],
                "phenotype": ann["phenotype"],
                "notes": ann["additional_notes"],
            }
        )

    obs_df = pd.DataFrame(obs_data).reset_index(drop=True)
    obs_df["tissue_type"] = pd.Categorical(obs_df["tissue_type"])
    obs_df["phenotype"] = pd.Categorical(obs_df["phenotype"])
    obs_df["region"] = region_labels_name

    uns_metadata = {}
    if existing_table is not None:
        existing_adata = getattr(existing_table, "adata", None)
        if existing_adata is not None and hasattr(existing_adata, "uns"):
            uns_metadata.update(
                {k: v for k, v in existing_adata.uns.items() if k != "spatialdata_attrs"}
            )

    adata = ad.AnnData(
        X=np.zeros((len(obs_df), 0)),
        obs=obs_df,
        uns=uns_metadata,
    )
    table_model = TableModel.parse(
        adata,
        region=region_labels_name,
        region_key="region",
        instance_key="region_id",
    )
    sdata.tables[table_name] = table_model

    print(f"Exported {len(unique_labels)} regions to SpatialData")
    return sdata


def link_cells_to_regions(
    sdata,
    cell_labels_name="instanseg_cell",
    region_labels_name="tissue_regions",
    cell_table_name="instanseg_table",
):
    """Assign each cell to a tissue region via centroid lookup.

    Uses cell centroids instead of full-mask majority voting, which avoids
    materializing both masks simultaneously and keeps memory usage minimal.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object with both cell and region segmentations.
    cell_labels_name : str
        Name of the cell labels layer.
    region_labels_name : str
        Name of the region labels layer.
    cell_table_name : str
        Name of the cell table.
    """
    print("Linking cells to regions...")

    # Keep region mask lazy; we only need point lookups.
    region_mask = sd.get_pyramid_levels(
        sdata.labels[region_labels_name], n=0
    ).squeeze()
    region_mask = _extract_data(region_mask)

    # Detect if region mask was saved at a lower resolution than the cell mask.
    # Compare against image or cell mask shape to infer downsample factor.
    downsample = 1
    if cell_labels_name in sdata.labels:
        cell_ref = sd.get_pyramid_levels(
            sdata.labels[cell_labels_name], n=0
        ).squeeze()
        cell_ref = _extract_data(cell_ref)
        cell_shape = cell_ref.shape[-2:]
        region_shape = region_mask.shape[-2:]
        if cell_shape[0] > region_shape[0] * 1.5:
            downsample = round(cell_shape[0] / region_shape[0])
            print(f"  Region mask is {downsample}x downsampled relative to cell mask")

    cell_table = sdata.tables[cell_table_name]
    region_table = sdata.tables["tissue_regions_table"]

    # Build a quick lookup from region_id -> metadata.
    region_meta: dict[int, dict] = {}
    for _, row in region_table.obs.iterrows():
        rid = int(row["region_id"])
        region_meta[rid] = {
            "tissue_type": row.get("tissue_type", "Unknown"),
            "phenotype": row.get("phenotype", ""),
            "notes": row.get("notes", ""),
        }

    # Use centroids for assignment (memory-efficient).
    obs = cell_table.obs
    if "centroid_y" in obs.columns and "centroid_x" in obs.columns:
        cy = obs["centroid_y"].values.astype(int)
        cx = obs["centroid_x"].values.astype(int)

        # Point lookup: scale centroids to region mask resolution.
        cy_scaled = np.clip(cy // downsample, 0, mask_shape[0] - 1)
        cx_scaled = np.clip(cx // downsample, 0, mask_shape[1] - 1)

        if _is_dask_array(region_mask):
            region_ids = region_mask.vindex[cy_scaled, cx_scaled].compute()
        else:
            region_ids = np.asarray(region_mask)[cy_scaled, cx_scaled]
        region_ids = np.asarray(region_ids, dtype=int)
    else:
        # Fallback: full mask majority vote (original behavior).
        print("  Centroids not found, falling back to full-mask majority vote...")
        cell_mask = np.asarray(
            _extract_data(
                sd.get_pyramid_levels(sdata.labels[cell_labels_name], n=0).squeeze()
            )
        )
        region_mask_np = np.asarray(region_mask)

        cell_ids = obs["label"].values
        region_ids = np.zeros(len(cell_ids), dtype=int)
        for i, cid in enumerate(cell_ids):
            pixels = cell_mask == cid
            overlapping = region_mask_np[pixels]
            overlapping = overlapping[overlapping != 0]
            if len(overlapping) > 0:
                region_ids[i] = int(np.bincount(overlapping).argmax())

    # Map region IDs to metadata columns.
    tissue_types = []
    phenotypes = []
    region_notes = []
    for rid in region_ids:
        meta = region_meta.get(int(rid), {})
        tissue_types.append(meta.get("tissue_type", "Unassigned" if rid == 0 else "Unknown"))
        phenotypes.append(meta.get("phenotype", ""))
        region_notes.append(meta.get("notes", ""))

    cell_table.obs["region_id"] = region_ids
    cell_table.obs["region_tissue_type"] = tissue_types
    cell_table.obs["region_phenotype"] = phenotypes
    cell_table.obs["region_notes"] = region_notes

    print(f"Linked {len(region_ids)} cells to regions")
    print(f"\nDistribution:")
    print(cell_table.obs["region_tissue_type"].value_counts())

    return sdata
