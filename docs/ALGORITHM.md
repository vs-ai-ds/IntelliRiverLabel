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

## Phase B — Curved Placement (optional wow factor)

Goal: place text along an internal path that follows flow direction.

### B1) Internal path extraction
From polygon P:
- Attempt to derive an internal path network (centerline-like).
- Select the main path (longest high-clearance path).

Important: if path extraction fails or is unstable, fall back to Phase A.

### B2) Segment windows
Slide windows along path with target arc length >= label length * 1.1.
For each window segment s:
- clearance_min(s) = min distance from sampled points on s to boundary(P)
- curvature(s) = sum |Δθ|
Prefer:
- high clearance_min
- low curvature
- stable direction

### B3) Render curved label
Render to SVG using a path definition and text-on-path.
If curved label violates padding/feasibility, fall back to Phase A.

## Debug overlays (required for judging)
- show P and safe region
- show sampled points or path segments colored by score
- show selected rectangle/path
- print metrics and feasibility reason for rejected candidates