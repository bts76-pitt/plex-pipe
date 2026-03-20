#!/usr/bin/env python
"""
Region Annotation Script for PlexPipe SpatialData cores.

Usage:
    # Annotate a specific core:
    python annotate_regions.py /path/to/Core_000.zarr

    # Annotate a specific core, then link cells to regions:
    python annotate_regions.py /path/to/Core_000.zarr --link-cells

    # List all available cores and pick one interactively:
    python annotate_regions.py --core-dir /path/to/cores/

    # Batch link cells to regions for all annotated cores (no napari):
    python annotate_regions.py --core-dir /path/to/cores/ --link-cells-only
"""

import argparse
import os
import sys
from pathlib import Path

import spatialdata as sd


def list_cores(core_dir: Path) -> list[Path]:
    """List all .zarr core directories, sorted."""
    cores = sorted(
        p for p in core_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.suffix == ".zarr"
    )
    return cores


def pick_core(core_dir: Path) -> Path:
    """Interactive core selection."""
    cores = list_cores(core_dir)
    if not cores:
        print(f"No .zarr directories found in {core_dir}")
        sys.exit(1)

    print(f"\nAvailable cores in {core_dir}:\n")
    for i, core in enumerate(cores):
        # Check if it already has tissue_regions
        has_annotations = ""
        try:
            sdata = sd.read_zarr(core)
            if "tissue_regions" in sdata.labels:
                n_regions = len(sdata.tables.get("tissue_regions_table", {}).obs) - 1 if "tissue_regions_table" in sdata.tables else "?"
                has_annotations = f"  [annotated, {n_regions} regions]"
            del sdata
        except Exception:
            has_annotations = "  [could not read]"

        print(f"  {i + 1}. {core.name}{has_annotations}")

    while True:
        choice = input(f"\nSelect core (1-{len(cores)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(cores):
                return cores[idx]
        except ValueError:
            pass
        print("Invalid selection, try again.")


def annotate_core(sd_path: Path, link_cells: bool = False):
    """Open napari for annotation, then optionally link cells."""
    from plex_pipe.ui.region_annotator import launch_region_annotation, link_cells_to_regions

    print(f"\nWorking on: {sd_path}")

    # Check what's available
    sdata_check = sd.read_zarr(sd_path)
    print(f"Available images: {list(sdata_check.images.keys())}")
    if "tissue_regions" in sdata_check.labels:
        print("Existing tissue regions found (will be loaded for editing)")
    if "instanseg_cell" in sdata_check.labels:
        print("Cell segmentation found (will be shown as reference layer)")
    del sdata_check

    # Launch napari (blocks until window is closed)
    # The save dialog runs automatically after closing napari
    sdata, widget = launch_region_annotation(str(sd_path))

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
            sdata.write(out_path, overwrite=True)
            print(f"Saved to {out_path}")

    return sdata


def batch_link_cells(core_dir: Path):
    """Link cells to regions for all cores that have both annotations and segmentation."""
    from plex_pipe.ui.region_annotator import link_cells_to_regions

    cores = list_cores(core_dir)
    print(f"\nScanning {len(cores)} cores for cell linkage...\n")

    for core_path in cores:
        try:
            sdata = sd.read_zarr(core_path)
        except Exception as e:
            print(f"  {core_path.name}: could not read ({e})")
            continue

        has_regions = "tissue_regions" in sdata.labels
        has_cells = "instanseg_cell" in sdata.labels
        has_table = "instanseg_table" in sdata.tables

        if not has_regions:
            print(f"  {core_path.name}: no tissue regions, skipping")
            continue
        if not has_cells or not has_table:
            print(f"  {core_path.name}: no cell segmentation/table, skipping")
            continue

        # Check if already linked
        if "region_tissue_type" in sdata.tables["instanseg_table"].obs.columns:
            print(f"  {core_path.name}: already linked, skipping")
            continue

        print(f"  {core_path.name}: linking cells to regions...")
        sdata = link_cells_to_regions(
            sdata=sdata,
            cell_labels_name="instanseg_cell",
            region_labels_name="tissue_regions",
            cell_table_name="instanseg_table",
        )

        out_path = core_path.parent / f"{core_path.stem}_linked.zarr"
        sdata.write(out_path, overwrite=True)
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
        help="Directory containing .zarr cores (for interactive selection or batch ops)",
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
            "/Volumes/HSIT-Stallaert-Lab/data_analysis/Lee-Oesterreich/NSR7649_Analysis/sdata/cores"
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

    annotate_core(sd_path, link_cells=args.link_cells)


if __name__ == "__main__":
    main()
