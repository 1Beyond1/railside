# Changelog

All notable changes to `railside` are documented in this file.

## [0.1.1] - 2026-11-26

### Security
- **Unified Path-Traversal Defense (`io.trip_dir`, `agent`, `db`, `report`)**: Enforce `TRIP_RE` (`^[A-Za-z0-9_-]{1,32}$`) and `MID` containment inside `io.trip_dir`; route invalid `trip` arguments in `Tools.call` to a safe `_invalid` audit bucket without creating attacker-controlled directories; guard `report.write_report` and `agent.write_agent_report` against out-of-bounds directory writes and `shutil.rmtree`.
- **FastAPI `POST /api/run` Hardening (`app/server.py`, `app/__init__.py`)**: Reject Windows UNC paths (`\\` / `//` prefixes) in `src` and `weights`; restrict `weights` to `.pt`/`.onnx` files inside `models/` or `RAILSIDE_MODELS_DIR`; require `RAILSIDE_APP_TOKEN` (`Authorization: Bearer` or `X-Railside-Token`) on non-loopback `--host` bindings and reject unauthenticated write requests with explicit startup warnings.
- **Placeholder Sanitization**: Replace private directory example in `app/static/index.html` with a generic path (`D:\railside_data\T03\camImg`).

### Fixed
- **MCP `stdio` & Pipeline Reliability (`pipeline.py`, `agent.py`)**: Add `log_fn` to `pipeline.run_trip` so MCP `run_trip` never pollutes `stdout`; validate `Path(src).is_dir()` before `archive_previous(trip)`; allow `run_trip` and `run_step(step="ingest")` for new trip IDs; remove unsupported `--template` from `calib`, avoid `--template None`, forward `--template` to `report`, propagate `RAILSIDE_DATA`, use `shutil.move` for archiving, and omit tool schemas in `write_agent_report` to prevent silent offline fallback.
- **Template-Trip Dual-Pane Sync & Frontend Fixes (`app/static/app.js`, `app/data.py`)**: Introduce `effCmpMode()` so viewing template trip `T01` on first load preserves dual-pane viewport sync and crosshair alignment; render `diff_area` as signed percentages; remove broken `file://` `ocr.html` link fallback; include `trip.json` `mtime_ns` in `_crop` LRU cache key and guard out-of-bounds crop boxes; fix overview `maxCar` to include `foreign`, use horizontal `sx` instead of `sy` for alignment mm estimate, prefer `f.get("bbox_xy")` for foreign alarms in Web/report, and protect printable reports against missing car metadata or missing crop images.
- **Algorithm, Scoring & Memory Fixes (`compare.py`, `eval/score.py`, `frames.py`, `match.py`, `calib.py`, `align.py`, `query.py`)**: Fix segmentation `diff_area` scaling from `s_amap ** 2` to linear `s_amap`; mask `missing` child parts in `find_foreign` to prevent double-reporting; write `bbox_xy` to `compare.json` items and fall back to `parts.json` in `eval/score.py` so part false positives are accurately counted; add vertical bounds clamping and streaming chunked downsampling in `Trip.region`; fix `match.aligned_region` `y_hi > y_lo` and tier-3 parent `pfound` writeback; guard `calib.refine_cap` against zero-division; validate `<2` valid blocks, filter non-finite correlations, and use `amap1.dy_at_x` in `align.py` pass-2; fix `query.part_history` `type=NULL` parameter mismatch and enforce SQLite `set_progress_handler` timeout and row cap in `query.sql`.
- **Web Background Job Concurrency (`app/jobs.py`, `db/__init__.py`, `io.py`)**: Protect `Job.lines` with per-job `threading.Lock`; fix `break` indentation in `partial` status detection; enforce per-trip exclusion (`409 Conflict`) for concurrent `run` jobs; add `DELETE /api/jobs/{id}` cancellation and FastAPI lifespan shutdown cleanup; use unique PID/TID/UUID temp files with locked retries in `io.write_json` and `db.build`.

### Changed
- **Dependency Cleanup**: Remove 4 unused runtime dependencies (`scipy`, `scikit-image`, `matplotlib`, `tifffile`) from `pyproject.toml` and refresh `uv.lock`.
- **CI & Test Coverage**: Refactor `源码/tests/test_app.py` to synthetic fixtures so all Web API and offline export tests execute without skipping; add regression tests covering MCP, security boundaries, algorithm fixes, concurrency, and frontend checks; add `node --check` and Windows matrix testing to GitHub Actions CI.

## [0.1.0] - 2026-11-23

### Added
- **End-to-End Inspection Pipeline (`railside run`)**: Unified CLI covering frame ingest & QC (`ingest`), full-train stitching (`stitch`), scale calibration (`calib`), template building (`template` / `label`), multi-anchor piecewise alignment (`align`), part matching (`match`), optional segmentation hooks (`train` / `segment`; no model weights distributed), SQLite storage & query (`db` / `query`), multi-criteria defect comparison (`compare`), and standalone HTML reporting (`report`).
- **Dual-Pane Template Comparison Web UI (`railside app`)**:
  - Synchronized upper/lower OpenSeadragon deep-zoom comparison (`本趟识别 vs 标准模板`) driven by piecewise `align.json` coordinate mapping with exact pixel-scale sync, `minPixelRatio=1.0` crisp tile rendering, and draggable split divider.
  - Real-time synced crosshair (`联动准星`) across both panes, 12-bogie sequential stepper, full-part search & filtering, and collapsible sidebar.
  - High-resolution comparison lightbox supporting **Curtain Slider (`卷帘对比`)**, **Side-by-Side (`并排对比`)**, **Hold-to-Flip (`按住切换`)**, and **Pixel Difference Highlight (`差异高亮`)** for both parts and foreign-object alarms.
  - Inspection overview dashboard (`巡检概览`) with KPI cards, per-car/per-type charts, manual review workflow (`人工复核`), and one-click printable/PDF inspection report (`巡检报告`) reflecting per-car manual review statuses.
  - Self-contained read-only offline bundle export (`railside app --export`) bundling both target trip and reference template DZI tiles.
- **Zero-Data Synthetic Demo (`railside demo`)**: Generates a 100% synthetic, privacy-free 3-car dataset (template `T01` + inspection trip `T02` with missing, rotated, shifted, and foreign-object alarms) in seconds for immediate evaluation.
- **Agent & MCP Integration (`railside agent` / `railside mcp`)**: Exposes inspection queries, alarm summaries, and pipeline execution over Model Context Protocol (`stdio`).
- **CI & Automated Testing**: GitHub Actions workflow running 167 automated unit and end-to-end tests on Ubuntu with `uv`.
