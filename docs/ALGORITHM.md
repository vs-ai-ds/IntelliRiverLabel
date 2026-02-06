# Algorithm

The river geometry input is a polygon in pt units. The label must be fully inside with padding. If impossible, place outside.

## Phase A — Robust Straight Placement (must succeed)

### A1) Build safe region
Given river polygon P and padding d:
- safe = P.buffer(-d)
If safe is empty:
- reduce d (down to a min), else declare internal infeasible

### A2) Generate candidate points
Goal: find points in the widest/most visible parts.
Method:
- Sample N points inside safe
  - Prefer: rejection sampling in bbox + inside check
  - Optional improvement: Poisson-like spacing (grid + jitter) for coverage

For each point p:
- clearance(p) = distance(p, boundary(safe))
Keep top K by clearance.

### A3) Propose angles
Compute dominant direction using PCA of polygon boundary points.
Test angle set:
- base angle (PCA axis)
- base ± 15°, ± 30° (configurable)
Also enforce uprightness: rotate by 180° if upside-down.

### A4) Label rectangle fit test
Compute label size in pt (w_pt, h_pt) using font metrics.
Build oriented rectangle R(p, angle, w_pt, h_pt).
Feasible if:
- R is fully contained in safe (with tolerance)
Score boost if it fits with margin.

### A5) Score and select
Heuristic score combines:
- + clearance (wider regions)
- + fit margin (how comfortably it fits)
- + centering (avoid ends if polygon is elongated)
- - angle penalty (optional readability constraint)

Pick best feasible candidate. If none feasible -> Phase A fails (rare) -> external placement.

## Phase B — Curved Placement (optional)

Goal: place text along an internal path that follows flow direction.

### B1) Internal path approximation (cross-section midpoints)
From safe polygon (P inset by padding), build a centerline without random sampling:
- Compute dominant axis using PCA angle from a simplified polygon.
- Along the river bounds on that axis, take a fixed number of slices (e.g. 80–200).
- For each slice position t: create a long line perpendicular to the axis; intersect with the safe polygon.
- From the intersection (LineString or MultiLineString), take the longest segment and collect its midpoint.
- Sort midpoints by t (order preserved), then smooth with a moving-average window (config).
- Return the LineString of smoothed midpoints if count ≥ PATH_MIN_POINTS, else None (fall back to Phase A).
Deterministic: slice count and all geometry depend only on the polygon, not random sampling.

### B2) Path window selection
Required window length >= label_width_pt * CURVE_FIT_MARGIN (e.g. 1.1).
- Slide windows along path by arclength.
- For each window: clearance_min = min distance to polygon boundary; curvature proxy = sum |Δθ|.
- Choose window with best clearance and low curvature.

### B3) Validation and output
- Validate min clearance along window >= padding_pt + CURVE_EXTRA_CLEARANCE_PT.
- Path length >= label_width_pt * 1.1.
- Produce PlacementResult with mode=phase_b_curved, path_pt (downsampled).
- Render to SVG (textPath, halo). If any step fails, return None and fall back to Phase A.

## Collision avoidance (multi-label)
When placing multiple labels on one river:
- **Occupied geometry**: Start with existing obstacles (optional). After each successful placement, add that label’s collision geometry to a union (occupied).
- **Collision geometry**: Phase A = buffered bbox rectangle; Phase B = buffered path polygon (font height/2 + padding).
- **Scoring**: Candidates that intersect occupied get a penalty (score − COLLISION_WEIGHT × normalized overlap). If overlap > COLLISION_MAX_AREA (pt²), the candidate is rejected.
- **Ordering**: Place longer (wider) labels first, tie-break by label text, so later shorter labels have more room.

## Multi-label ordering
Labels are sorted by measured width (descending), then by text. This reduces failures when the river is narrow: long labels get priority.

## Zoom bucket scaling
To get consistent placement across zoom levels, run placement at several **zoom buckets** (e.g. 10, 12, 14). For each bucket (or bucket index):
- `font_pt = base_font × (1 + ZOOM_FONT_SCALE_FACTOR × bucket_index)`
- `padding_pt = base_padding × (1 + ZOOM_PADDING_SCALE_FACTOR × bucket_index)`
Outputs can be written per bucket under `reports/<run_name>/zoom_<bucket>/` with a `zoom_index.json` summarizing metrics and optional “best bucket” choice.

## Batch processing flow
1. Input: directory of .wkt files or a manifest CSV (path, labels).
2. For each case: load geometry → preprocess → run layout (single or multi-label) → write placement(s).json, before/after/debug, run_metadata.
3. Write **index.csv** with columns: case_id, geometry_source, labels, mode_used, n_labels, success_count, mean_min_clearance, mean_fit_margin, collisions_detected, duration_ms, warnings_count.
4. All under `reports/batch_<run_name>/` and `cases/<case_id>/`.

## Debug overlays (required for judging)
- show P and safe region
- show sampled points or path segments colored by score
- show selected rectangle/path
- **Multi-label**: show occupied areas (semi-transparent hatch) and label collision geometry outlines
- print metrics and feasibility reason for rejected candidates