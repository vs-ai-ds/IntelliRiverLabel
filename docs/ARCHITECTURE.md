# Architecture

## 1) High-level Overview
Pipeline:
1. Load geometry (WKT/GeoJSON) -> normalize -> polygon(s)
2. Preprocess (fix validity, simplify, inset safe polygon)
3. Candidate generation
   - Phase A: interior points + angles
   - Phase B: internal path segments (centerline-like)
4. Feature extraction
5. Scoring (heuristic + optional learned ranking)
6. Placement selection + validation
7. Rendering (PNG + optional SVG) + metrics output
8. Streamlit UI

## 2) Modules
- app/core/io.py
  - Read WKT/GeoJSON
  - Validate geometry, fix with buffer(0) if needed
  - Supports Polygon, MultiPolygon, GeometryCollection (extract polygons)
- app/core/preprocess.py
  - Simplify polygon, normalize orientation
  - Create safe polygon with inward buffer (padding)
- app/core/candidates_a.py
  - Generate interior sample points
  - Propose angle candidates (PCA axis + jitter angles)
- app/core/candidates_b.py
  - Extract internal path and candidate segments
  - Choose segment windows matching label length
- app/core/features.py
  - Clearance metrics, fit ratio, stability, curvature proxy
- app/core/scoring.py
  - Heuristic score (always)
  - Optional learned score (model loaded from app/models)
- app/core/placement.py
  - Convert candidate -> placement params (x, y, angle, font_size, path)
- app/core/validate.py
  - Ensure label bbox/path lies inside safe polygon
  - External fallback only if internal infeasible
- app/core/render.py
  - PNG before/after/debug overlays
  - Optional SVG renderer for path-based label
- app/models/train.py
  - Synthetic data generation + model training (optional)
- app/models/registry.py
  - Load model artifact; version metadata
- app/ui/app.py
  - Streamlit: load default `assets/problem/Problem_1_river.wkt`, upload, debug, export

## 3) Extensibility Hooks
- Add obstacle layers and collision checks in validate.py
- Add multi-river batch mode in runner + UI
- Add zoom-bucket scaling (pt->px) and consistent typography in render.py