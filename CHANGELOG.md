# Changelog

All notable changes to `railside` are documented in this file.

## [Unreleased] - 0.1.0-dev

### Added
- **End-to-End Inspection Pipeline (`railside run`)**: Unified CLI covering frame ingest & QC (`ingest`), full-train stitching (`stitch`), scale calibration (`calib`), template building (`template` / `label`), multi-anchor piecewise alignment (`align`), part matching (`match`), optional multi-scale YOLO-seg training/inference (`train` / `segment`), SQLite storage & query (`db` / `query`), multi-criteria defect comparison (`compare`), and standalone HTML reporting (`report`).
- **Dual-Pane Template Comparison Web UI (`railside app`)**:
  - Synchronized upper/lower OpenSeadragon deep-zoom comparison (`本趟识别 vs 标准模板`) driven by piecewise `align.json` coordinate mapping with exact pixel-scale sync and draggable split divider.
  - Real-time synced crosshair (`联动准星`) across both panes, 12-bogie sequential stepper, full-part search & filtering, and collapsible sidebar.
  - High-resolution comparison lightbox supporting **Curtain Slider (`卷帘对比`)**, **Side-by-Side (`并排对比`)**, **Hold-to-Flip (`按住切换`)**, and **Pixel Difference Highlight (`差异高亮`)** for both parts and foreign-object alarms.
  - Inspection overview dashboard (`巡检概览`) with KPI cards, per-car/per-type charts, manual review workflow (`人工复核`), and one-click printable/PDF inspection report (`巡检报告`).
  - Self-contained read-only offline bundle export (`railside app --export`) bundling both target trip and reference template DZI tiles.
- **Zero-Data Synthetic Demo (`railside demo`)**: Generates a 100% synthetic, privacy-free 3-car dataset (template `T01` + inspection trip `T02` with missing, rotated, shifted, and foreign-object alarms) in seconds for immediate evaluation.
- **Agent & MCP Integration (`railside agent` / `railside mcp`)**: Exposes inspection queries, alarm summaries, and pipeline execution over Model Context Protocol (`stdio`).
- **CI & Automated Testing**: GitHub Actions workflow running 167 automated unit and end-to-end tests on Ubuntu with `uv`.
