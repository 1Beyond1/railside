[简体中文](README.md) | English

# railside: Side-View Visual Inspection for Metro Trains

`railside` processes raw side-view line-scan camera frames of passing metro trains to produce full-train stitched imagery, a coded part inventory, and per-part comparison alarms against a reference template train, along with a local web viewer and agent/MCP tooling.

## Pipeline

| Stage | Command | Description |
|---|---|---|
| Ingest & QC | `railside ingest` | Checks frame order, dropped frames, exposure, and blur |
| Stitch | `railside stitch` | Stitches consecutive frames into per-car panels (`train_panels`), writing `manifest.json` and 1/8-scale overview previews |
| Tile export | `railside tiles` | Exports overlapping tiles in trip coordinates for labeling, training, and inference |
| Calibration | `railside calib` | Computes mm/px scale from known carbody dimensions |
| Template train | `railside template` / `label` | Builds the template train: alignment anchors and part polygons (SAM 2.1 box-prompt pre-labeling + human review) |
| Alignment | `railside align` | Aligns a new trip to template-train coordinates via anchors |
| Segmentation | `railside train` / `segment` | Builds/trains a multi-scale YOLO-seg model from template labels and runs sliding-window inference across a trip (optional) |
| Part matching | `railside match` | Locates each expected part and assigns part codes and match scores |
| Database & query | `railside db` / `query` | Writes results into SQLite and queries by car, bogie, or part type |
| Comparison | `railside compare` | Detects missing parts, displacement, rotation, occlusion, and foreign-object alarms |
| Report | `railside report` | Generates a standalone per-trip HTML report (`index.html` + part crops) |
| End-to-end run | `railside run <raw-dir> --trip <trip-id>` | Runs the full pipeline from raw frames to database and HTML report |
| Web viewer | `railside app` | Local web UI for full-train DZI deep-zoom browsing, part details, and trip comparison (supports `--export` for read-only offline bundles) |
| Agent & MCP | `railside agent` / `mcp` | Exposes inspection queries, alarms, and pipeline runs as agent tools over MCP (stdio) |

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run railside --help
uv run pytest 源码/tests
```

`pyproject.toml` enables the `dev` and `dl` dependency groups by default (`ultralytics` is used for segmentation training/inference and SAM pre-labeling; model weights are not distributed in this repository). To install without deep-learning dependencies, run `uv sync --no-group dl`.

## Directory Layout

```text
源码/railside/   Python package; one module per pipeline stage, wired together in cli.py
源码/tests/      Test suite using synthetic data (no real train imagery required)
主数据/          Master part-type table and comparison thresholds (零件类型.csv)
数据/            Runtime data (raw frames, intermediate artifacts, outputs); excluded from Git
```

## Data Notice

Real train image datasets belong to their respective owners and are neither included in this repository nor covered by its license.

## License

This project is licensed under the GNU AGPL-3.0 — see [LICENSE](LICENSE). You may freely use, modify, and distribute it, provided that derivative works (including network services built on top of it) also release their source code under AGPL-3.0.

For closed-source or commercial licensing, please contact the author via GitHub.

Copyright (c) 2026 1Beyond1

`源码/railside/app/static/vendor/openseadragon/` bundles OpenSeadragon 6.1.1 under its own BSD-3-Clause license (see [LICENSE.txt](源码/railside/app/static/vendor/openseadragon/LICENSE.txt)).
