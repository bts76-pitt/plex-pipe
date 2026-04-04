# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands should be run from the repo root (`tools/plex-pipe/`).

**Install for development:**
```bash
uv sync --extra all-gpu    # GPU-enabled (includes segmentation, GUI)
uv sync --extra segmentation-cpu  # CPU-only segmentation
```

**Run tests:**
```bash
python -m pytest -v --color=yes --cov=plex_pipe --cov-report=xml:coverage.xml
python -m pytest tests/path/to/test_file.py -v  # single file
python -m tox                                    # full matrix (3.11–3.13)
```

**Lint and format:**
```bash
ruff check --fix src/
black src/
mypy src/
pre-commit run --all-files
```

**Docs:**
```bash
mkdocs serve
```

## Architecture

The package (`src/plex_pipe/`) implements a **stage-based pipeline** for analyzing whole-slide multiplexed immunofluorescence images.

### Pipeline Stages (`stages/`)

Stages run sequentially, each with a dedicated controller:

1. **`roi_definition/`** — detects/defines regions of interest from metadata
2. **`roi_preparation/`** — cuts ROI cores from whole-slide OME-TIFFs via `CoreCutter` → `CoreAssembler`; file access is abstracted behind a `FileAvailabilityStrategy` (`LocalFileStrategy` / `GlobusFileStrategy`)
3. **`resource_building/`** — runs registered `BaseOp` operations on SpatialData, builds multi-resolution pyramids, manages Zarr storage
4. **`quantification/`** — extracts morphological and intensity metrics into AnnData tables

### Registry + Processor Pattern (`ops/`)

All image operations (segmenters, mask builders, image enhancers) follow the same contract:

- Inherit from `BaseOp`; configuration via a nested `Params(ProcessorParamsBase)` Pydantic model
- Registered with `@register(kind, name)` — kinds are `mask_builder`, `object_segmenter`, `image_enhancer`
- Auto-discovered on import via `iter_modules()` in `ops/__init__.py`
- Instantiated via `build_processor(kind, name, **cfg)` factory

To add a new processor: create a file in `ops/`, define a `Params` class and a class decorated with `@register`, and it's automatically available in YAML configs.

### Configuration Layer (`config/`)

- `config_schema.py` — Pydantic v2 models: `GeneralSettings`, `RoiDefinitionSettings`, `RoiCuttingSettings`, `QuantTask`, `AnalysisConfig` (root)
- `config_loaders.py` — YAML loading with `${input}` placeholder expansion; `expand_pipeline()` fans out steps with list inputs
- `PipelineStep` is a dynamic union type built from the registry at import time

### Data Formats

- **Input:** OME-TIFF (Cell DIVE format)
- **Intermediate:** SpatialData backed by Zarr v2/v3
- **Output:** AnnData tables (scverse ecosystem compatible)

### Optional Components

- **GUI** (`ui/`): napari widgets for ROI selection, QC masking, region annotation — requires `gui` extra
- **Globus** (`io/globus.py`): remote institutional data transfer — requires `globus` extra
- **Segmentation** (`ops/object_segmenters.py`): Cellpose / InstanSeg — requires `segmentation-cpu` or `segmentation-gpu`

## Code Style

- Line length: 79 (black + ruff)
- Python ≥ 3.11; mypy strict mode enabled
- Ruff rules: E, F, W, UP, I, BLE, B, A, C4, ISC, G, PIE, SIM, NPY, ARG, PT, PD
- Logging via `loguru` throughout (not stdlib `logging`)
- Source layout: `src/plex_pipe/`; tests in `tests/`
- `conftest.py` mocks napari/qtpy for headless CI
