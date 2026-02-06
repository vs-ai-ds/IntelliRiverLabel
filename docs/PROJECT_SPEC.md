# IntelliRiverLabel — Project Spec 

## 1) Problem
Given:
- River geometry: `river.wkt` (unit: pt) — may consist of multiple parts
- A text label (e.g., "ELBE") with a specific font size in pt (e.g., 12 pt)

Task:
Place the river name once inside the river geometry in a cartographically appealing way.

Requirements:
1) Text must be completely inside the river boundary with proper padding (no touching edges)
2) Position text for optimal readability — centered in the widest/most visible part of the river
3) Only if no space inside: text may be placed outside the geometry

Optional enhancements:
- Text rotation to align with river flow direction
- Curve-following along river centerline (advanced)

Source references (workspace layout):
- Problem statement PDF: `assets/problem/Problem_1_river.wkt`
- Blurb image: `assets/problem/problem_1.pdf`
- Geometry: `assets/problem/problem_1_blurb.jpeg`

## 2) Project Goal
Build a reliable, extensible Python system that:
- Places a river label inside a polygon river with safe padding
- Produces a clear before/after demo + debug overlays
- Exports a machine-readable placement JSON for future integration
- Supports a curved-text mode with safe fallback to straight placement

## 3) Inputs
Required:
- River geometry: `docs/assets/problem/Problem_1_river.wkt` (repo-relative; Polygon or MultiPolygon, unit: pt)
- `label_text` (string)
- `font_size_pt` (number)

Optional:
- `padding_pt` (number, default from config)
- `mode` (phase_a / phase_b)
- `render_width_px`, `render_height_px` (for output images)
- `seed` (for deterministic sampling)

## 4) Outputs
All under `reports/<run_name>/`:
- `placement.json` (see `docs/PLACEMENT_SCHEMA.md`)
- `before.png`, `after.png`
- `debug.png` (safe polygon, candidate points/segments, chosen placement)
- `run_metadata.json`
- Optional: `after.svg` (for curved text path)

## 5) Success Criteria
Functional:
- Places label inside polygon with padding for provided test geometry
- Falls back outside only if no feasible internal placement exists

Quality:
- Outperforms centroid baseline on clearance/fit metrics on a curated set
- Produces visually clean label placement (halo, upright text, minimal distortion)

Engineering:
- Clean module boundaries (`core` vs `ui`)
- Config-driven, testable core logic
- Easy to add future features (collisions, multi-river batch)

## 6) Scope Control
In scope:
- Polygon/MultiPolygon handling
- Straight label placement + rotation
- Optional curved placement using an internal path
- **Multi-label layout** with **collision avoidance** (occupied geometry, penalty/reject)
- **Batch mode**: directory of .wkt or manifest CSV; output index.csv + cases/
- **Zoom buckets**: consistent placement/scale by zoom (scaled font/padding per bucket)
- **Evaluation families** (straight_wide, curved_narrow, etc.) and **leaderboard** export (CSV/JSON)
- Debug visualization + metrics + export

Out of scope:
- Full multi-layer map labeling engine
- Heavy model training / deep learning
- Mobile app build

## 7) Supported geometry
- **POLYGON**, **MULTIPOLYGON**, or **GEOMETRYCOLLECTION** containing polygons (unit: pt).
- Multi-part: user may select a component (Auto = best by safe area, or index 0, 1, 2…).

## 8) Known limitations
- Phase B (curved) requires a sufficiently long centerline path; narrow or very irregular polygons may fall back to Phase A.
- Multi-label collision uses a 2D occupied union; very dense labels may fail placement for later labels.
- Batch and evaluation runs write under `reports/`; do not commit these artifacts.