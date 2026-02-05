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

### B1) Internal path approximation
From safe polygon (P inset by padding):
- Sample interior points (Phase A sampling), keep top-K by clearance.
- Compute PCA dominant axis of safe polygon boundary.
- Project points onto PCA axis and sort by projected coordinate.
- Smooth the ordered points (moving average) to form a polyline.
- This polyline is the internal path for text-on-path.

If path build fails or path too short, return None and fall back to Phase A.

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

## Debug overlays (required for judging)
- show P and safe region
- show sampled points or path segments colored by score
- show selected rectangle/path
- print metrics and feasibility reason for rejected candidates