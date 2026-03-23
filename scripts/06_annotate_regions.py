#!/usr/bin/env python
"""
Region Annotation Script for PlexPipe SpatialData cores.

Usage:
    # Annotate a specific core:
    python annotate_regions.py /path/to/Core_000.zarr

    # Annotate a specific core, then link cells to regions:
    python annotate_regions.py /path/to/Core_000.zarr --link-cells

    # List all available cores and pick one interactively:
    python annotate_regions.py --core-dir /path/to/Lee-Oesterreich/

    # Only load specific channels (faster startup, less memory):
    python annotate_regions.py /path/to/Core_000.zarr --channels DAPI CK818 CK14 ECad

    # Batch link cells to regions for all annotated cores (no napari):
    python annotate_regions.py --core-dir /path/to/cores/ --link-cells-only
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import spatialdata as sd

from plex_pipe.io.spatialdata_zarr import read_spatialdata_zarr


def list_cores(core_dir: Path) -> list[Path]:
    """List all .zarr core/ROI directories, searching common PlexPipe layouts.

    Handles both structures:
        <Analysis>/sdata/cores/Core_XXX.zarr
        <Analysis>/rois/ROI_XXX.zarr

    If core_dir points directly to a cores/ or rois/ folder, lists its contents.
    If core_dir points to an Analysis folder, searches for sdata/cores/ and rois/.
    If core_dir is a parent of multiple Analysis folders, searches all of them.
    """
    zarr_dirs: list[Path] = []

    # Case 1: core_dir itself contains .zarr directories
    local_zarrs = sorted(
        p for p in core_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.suffix == ".zarr"
    )
    if local_zarrs:
        zarr_dirs.extend(local_zarrs)
        return sorted(set(zarr_dirs), key=lambda p: p.name)

    # Case 2: core_dir is an Analysis folder or parent -- search recursively
    search_patterns = [
        "sdata/cores",
        "rois",
        "*/sdata/cores",
        "*/rois",
    ]
    for pattern in search_patterns:
        for subdir in sorted(core_dir.glob(pattern)):
            if not subdir.is_dir():
                continue
            for p in sorted(subdir.iterdir()):
                if p.is_dir() and not p.name.startswith(".") and p.suffix == ".zarr":
                    zarr_dirs.append(p)

    return sorted(set(zarr_dirs), key=lambda p: str(p))


def pick_core(core_dir: Path) -> Path:
    """Interactive core selection."""
    cores = list_cores(core_dir)
    if not cores:
        print(f"No .zarr directories found in or under {core_dir}")
        sys.exit(1)

    print(f"\nAvailable cores/ROIs (searched from {core_dir}):\n")
    for i, core in enumerate(cores):
        # Show the parent context so you know which sample it's from
        try:
            rel = core.relative_to(core_dir)
        except ValueError:
            rel = core.name

        # Check if it already has tissue_regions
        has_annotations = ""
        try:
            sdata = read_spatialdata_zarr(core)
            if "tissue_regions" in sdata.labels:
                n_regions = "?"
                if "tissue_regions_table" in sdata.tables:
                    n_regions = len(sdata.tables["tissue_regions_table"].obs) - 1
                has_annotations = f"  [annotated, {n_regions} regions]"
            del sdata
        except Exception:
            has_annotations = "  [could not read]"

        print(f"  {i + 1}. {rel}{has_annotations}")

    while True:
        choice = input(f"\nSelect core (1-{len(cores)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(cores):
                return cores[idx]
        except ValueError:
            pass
        print("Invalid selection, try again.")


def annotate_core(sd_path: Path, link_cells: bool = False, channels: list[str] | None = None):
    """Open napari for annotation, then optionally link cells."""
    from plex_pipe.ui.region_annotator import (
        launch_region_annotation,
        link_cells_to_regions,
        sync_spatialdata_to_store,
        zarr_store_copy_ignore,
    )

    print(f"\nWorking on: {sd_path}")

    # Check what's available (skip broken raster subgroups from interrupted saves)
    sdata_check = read_spatialdata_zarr(sd_path)
    if not sdata_check.images:
        print(
            "ERROR: No readable image layers in this zarr. The store may be corrupted "
            "(e.g. incomplete write). Try the original core, or remove broken groups "
            "under images/ or labels/ that lack OME-Zarr metadata.",
            file=sys.stderr,
        )
        sys.exit(1)
    if "annotated" in sd_path.name.lower() and "tissue_regions" not in sdata_check.labels:
        print(
            "\nNote: No readable `tissue_regions` layer in this file (common after a failed "
            "save). Images loaded; you can annotate again. To clean up: delete "
            "`labels/tissue_regions` inside the .zarr, or re-copy from the source core.\n"
        )
    print(f"Available images: {list(sdata_check.images.keys())}")
    if "tissue_regions" in sdata_check.labels:
        print("Existing tissue regions found (will be loaded for editing)")
    if "instanseg_cell" in sdata_check.labels:
        print("Cell segmentation found (will be shown as reference layer)")
    del sdata_check

    # Launch napari (blocks until window is closed)
    # The save dialog runs automatically after closing napari
    sdata, widget = launch_region_annotation(str(sd_path), channels=channels)

    # Optionally link cells to regions
    if link_cells and "instanseg_cell" in sdata.labels and "tissue_regions" in sdata.labels:
        print("\nLinking cells to regions...")
        sdata = link_cells_to_regions(
            sdata=sdata,
            cell_labels_name="instanseg_cell",
            region_labels_name="tissue_regions",
            cell_table_name="instanseg_table",
        )

        if "instanseg_table" in sdata.tables:
            cell_table = sdata.tables["instanseg_table"]
            print("\nCell-to-region assignments (first 10):")
            print(cell_table.obs[["region_id", "region_tissue_type"]].head(10))
            print("\nDistribution of cells by tissue type:")
            print(cell_table.obs["region_tissue_type"].value_counts())

        # Save again with cell linkage
        save_choice = input("\nSave updated cell linkage? (y/n): ").strip().lower()
        if save_choice == "y":
            out_path = sd_path.parent / f"{sd_path.stem}_annotated.zarr"
            if out_path.exists():
                confirm = input(f"{out_path} already exists. Overwrite? (y/n): ").strip().lower()
                if confirm != "y":
                    print("Skipped saving cell linkage.")
                    return sdata
                shutil.rmtree(out_path)
            # Avoid full ``sdata.write`` (rewrites images; breaks Zarr v2 vs NGFF 0.5 mismatch).
            shutil.copytree(sd_path, out_path, ignore=zarr_store_copy_ignore)
            labels = ("tissue_regions",) if "tissue_regions" in sdata.labels else ()
            tables = tuple(
                t
                for t in ("tissue_regions_table", "instanseg_table")
                if t in sdata.tables
            )
            sync_spatialdata_to_store(sdata, out_path, labels=labels, tables=tables)
            print(f"Saved to {out_path}")

    return sdata


def batch_link_cells(core_dir: Path):
    """Link cells to regions for all cores that have both annotations and segmentation."""
    from plex_pipe.ui.region_annotator import link_cells_to_regions, sync_spatialdata_to_store, zarr_store_copy_ignore

    cores = list_cores(core_dir)
    print(f"\nScanning {len(cores)} cores for cell linkage...\n")

    for core_path in cores:
        try:
            rel = core_path.relative_to(core_dir)
        except ValueError:
            rel = core_path.name

        try:
            sdata = read_spatialdata_zarr(core_path)
        except Exception as e:
            print(f"  {rel}: could not read ({e})")
            continue

        has_regions = "tissue_regions" in sdata.labels
        has_cells = "instanseg_cell" in sdata.labels
        has_table = "instanseg_table" in sdata.tables

        if not has_regions:
            print(f"  {rel}: no tissue regions, skipping")
            continue
        if not has_cells or not has_table:
            print(f"  {rel}: no cell segmentation/table, skipping")
            continue

        # Check if already linked
        if "region_tissue_type" in sdata.tables["instanseg_table"].obs.columns:
            print(f"  {rel}: already linked, skipping")
            continue

        print(f"  {rel}: linking cells to regions...")
        sdata = link_cells_to_regions(
            sdata=sdata,
            cell_labels_name="instanseg_cell",
            region_labels_name="tissue_regions",
            cell_table_name="instanseg_table",
        )

        out_path = core_path.parent / f"{core_path.stem}_linked.zarr"
        if out_path.exists():
            shutil.rmtree(out_path)
        shutil.copytree(core_path, out_path, ignore=zarr_store_copy_ignore)
        sync_spatialdata_to_store(sdata, out_path, tables=("instanseg_table",))
        print(f"    Saved to {out_path}")

    print("\nBatch linking complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate tissue regions in SpatialData cores using napari.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "core_path",
        nargs="?",
        type=Path,
        help="Path to a specific .zarr core to annotate",
    )
    parser.add_argument(
        "--core-dir",
        type=Path,
        default=None,
        help="Directory containing .zarr cores, or parent Analysis directory",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Only load these image channels (e.g. --channels DAPI CK818 CK14 ECad)",
    )
    parser.add_argument(
        "--link-cells",
        action="store_true",
        help="After annotation, link cells to tissue regions",
    )
    parser.add_argument(
        "--link-cells-only",
        action="store_true",
        help="Batch link cells to regions for all annotated cores (no napari)",
    )

    args = parser.parse_args()

    # Batch cell linkage mode
    if args.link_cells_only:
        if args.core_dir is None:
            print("--link-cells-only requires --core-dir")
            sys.exit(1)
        batch_link_cells(args.core_dir)
        return

    # Determine which core to work on
    if args.core_path is not None:
        sd_path = args.core_path
    elif args.core_dir is not None:
        sd_path = pick_core(args.core_dir)
    else:
        # Default path
        default_dir = Path(
            "/Volumes/HSIT-Stallaert-Lab/data_analysis/Lee-Oesterreich"
        )
        if default_dir.exists():
            sd_path = pick_core(default_dir)
        else:
            print("Please provide a core path or --core-dir")
            parser.print_help()
            sys.exit(1)

    if not sd_path.exists():
        print(f"Path does not exist: {sd_path}")
        sys.exit(1)

    annotate_core(sd_path, link_cells=args.link_cells, channels=args.channels)


if __name__ == "__main__":
    main()
