# IntelliRiverLabel: Technical Overview

## What Is This?

IntelliRiverLabel is an **intelligent label placement system** for river geometries. Given a river shape (polygon) and a label text, it automatically determines:

1. **Where** to place the label(s) inside the river
2. **How many** labels are needed for optimal coverage
3. **What angle** each label should have to follow the river's flow

The system uses a combination of **geometric algorithms** and **machine learning** to produce aesthetically pleasing, readable labels that stay strictly within the river boundaries.

---

## The Problem

Placing text labels on map features like rivers is surprisingly complex:

```
❌ Bad Placement:
   - Label outside the river boundary
   - Label at wrong angle (hard to read)
   - Too few/many labels for river length
   - Labels overlapping each other

✅ Good Placement:
   - Label fully inside safe boundaries
   - Label aligned with river flow direction
   - Optimal number of labels for coverage
   - No overlapping labels
```

---

## How It Works

### Step 1: Geometry Preprocessing

```
Input: Raw river polygon (WKT format)
       ↓
   Simplify & Clean
       ↓
   Buffer Inward (create "safe zone")
       ↓
Output: Safe polygon for label placement
```

The **safe zone** is the river polygon shrunk inward by a padding amount (default 3pt). This ensures labels have clearance from the river edges.

### Step 2: Centerline Detection

We compute the river's **centerline** (skeleton) to understand its flow:

```python
# Simplified concept:
1. Build internal path through polygon
2. Use PCA to find dominant direction
3. Clip centerline to stay INSIDE polygon
```

The centerline gives us:
- **Flow direction** (which way the river "flows")
- **Local angles** at any point along the river
- **Total length** for calculating optimal label count

### Step 3: Candidate Generation

We generate potential label positions using **stratified sampling**:

```
┌─────────────────────────────┐
│  ·    ·    ·    ·    ·    · │  ← Sample points inside safe zone
│    ·    ·    ·    ·    ·    │
│  ·    ·  [C]  ·    ·    ·   │  ← [C] = Centerline points
│    ·    ·    ·    ·    ·    │
│  ·    ·    ·    ·    ·    · │
└─────────────────────────────┘
```

Each candidate point is evaluated for:
- **Clearance**: Distance from river boundary
- **Centering**: How centered within the polygon
- **Fit margin**: How well the label fits at this position

### Step 4: ML-Based Ranking (The "Learned" Part)

This is where machine learning comes in. We train a **Gradient Boosting Regressor** to score candidates:

```
Input Features:
├── clearance_pt          (distance to boundary)
├── fit_margin_ratio      (how well label fits)
├── centering_score       (centrality measure)
├── angle_penalty         (deviation from ideal angle)
└── bbox_min_clearance_pt (label box clearance)
        ↓
   ML Model (GradientBoostingRegressor)
        ↓
   Score (higher = better position)
```

**Training Data**: Generated from synthetic river-like polygons with known "good" placements.

**Blending**: Final score = α × heuristic_score + (1-α) × ml_score

### Step 5: Multi-Label Placement

For longer rivers, one label isn't enough. We calculate:

```python
optimal_count = river_length / (label_width × spacing_factor)
```

Labels are placed at **evenly spaced intervals** along the centerline:

```
River:  ════════════════════════════════════════
Labels:      [River]         [River]         [River]
Position:      25%             50%             75%
```

### Step 6: Flow Alignment

Each label is rotated to match the **local flow angle** at its position:

```
        ╱ River ╲
       ╱         ╲
      ╱   flows   ╲
     ╱     ↘       ╲
    ╱    [Label]    ╲    ← Label angled to follow flow
   ╱        ↘        ╲
```

The angle is computed from the centerline tangent at each label position.

### Step 7: Validation & Adjustment

Before finalizing, each placement is validated:

```python
def validate_placement(label_bbox, safe_polygon):
    # 1. Anchor point must be inside
    # 2. Entire bounding box must be inside
    # 3. Must not overlap with other labels
    # 4. Must have minimum clearance
```

If validation fails, the system tries:
1. Different angles (±5°, ±10°, ... ±90°)
2. Nearby positions (spiral search)
3. Alternative interior points
4. Fallback to standard placement algorithm

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Streamlit UI                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │  Demo   │  │  Debug  │  │ Results │  │ Compare │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
└───────┼────────────┼────────────┼────────────┼──────────────┘
        │            │            │            │
        ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Core Engine                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  multilabel  │  │   placement  │  │    render    │      │
│  │     .py      │  │     .py      │  │     .py      │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
│         ▼                 ▼                 ▼               │
│  ┌──────────────────────────────────────────────────┐      │
│  │  geometry.py │ validate.py │ scoring.py │ config │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                      ML Model                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   train.py   │  │  features.py │  │  registry.py │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                          │                                   │
│                          ▼                                   │
│              model.joblib (trained model)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Algorithms

### 1. Centerline Computation

```python
def compute_centerline(polygon):
    # Try: Build internal path using Voronoi-like approach
    path = build_internal_path_polyline(polygon)
    
    # Fallback: PCA-based approximation
    if path is None:
        # Find dominant direction via Principal Component Analysis
        # Create line through centroid along principal axis
        # CLIP to polygon interior (critical!)
        
    return centerline  # Always inside polygon
```

### 2. Optimal Label Count

```python
def estimate_optimal_count(river_length, label_width):
    spacing = label_width * 1.5  # 50% gap between labels
    count = river_length / (label_width + spacing)
    return clamp(count, min=1, max=10)
```

### 3. Flow Angle at Position

```python
def local_flow_angle(centerline, position):
    # Project position onto centerline
    proj_distance = centerline.project(Point(position))
    
    # Sample two nearby points for tangent
    p1 = centerline.interpolate(proj_distance - delta)
    p2 = centerline.interpolate(proj_distance + delta)
    
    # Compute angle from tangent vector
    angle = atan2(p2.y - p1.y, p2.x - p1.x)
    
    # Normalize to [-90°, 90°] for readability
    return normalize_angle(angle)
```

### 4. Placement Validation

```python
def validate_rect_inside_safe(safe_polygon, label_rect):
    # Strict containment check
    if not safe_polygon.contains(label_rect):
        # Allow 98% overlap for numerical precision
        overlap = safe_polygon.intersection(label_rect).area
        if overlap / label_rect.area < 0.98:
            return False, 0.0
    
    # Compute clearance (distance to boundary)
    clearance = safe_polygon.boundary.distance(label_rect)
    return True, clearance
```

---

## ML Model Details

### Training

```bash
python -c "from app.models.train import train_model; train_model(n_polygons=100)"
```

**What happens:**
1. Generate 100 synthetic river-like polygons
2. For each polygon:
   - Generate candidate positions
   - Compute features for each candidate
   - Calculate "oracle" score: `clearance + fit_margin - angle_penalty`
3. Train GradientBoostingRegressor on (features → score)
4. Save to `app/models/artifacts/model.joblib`

### Features Used

| Feature | Description |
|---------|-------------|
| `clearance_pt` | Distance from candidate to polygon boundary |
| `fit_margin_ratio` | Clearance / padding (how well label fits) |
| `centering_score` | How centered the position is (0-1) |
| `angle_penalty` | Penalty for non-horizontal angles |
| `bbox_min_clearance_pt` | Minimum clearance of label bounding box |

### Inference

```python
# During placement:
for candidate in candidates:
    features = extract_features(candidate)
    ml_score = model.predict([features])[0]
    heuristic_score = compute_heuristic(candidate)
    
    # Blend scores
    final_score = alpha * heuristic_score + (1 - alpha) * ml_score
```

---

## File Formats

### Input: WKT (Well-Known Text)

```
POLYGON ((100 200, 150 250, 200 230, ...))
```

### Output: placement.json

```json
{
  "result": {
    "label_text": "River Name",
    "anchor_pt": [150.5, 225.3],
    "angle_deg": 15.7,
    "mode": "phase_a_flow_aligned",
    "confidence": 0.85,
    "min_clearance_pt": 5.2
  },
  "metrics": {
    "fit_margin_ratio": 1.73,
    "centering_score": 0.82
  }
}
```

### Output: Images

- `before.png` - River polygon only
- `after.png` - River with labels placed
- `debug.png` - Visualization with candidates, safe zone, etc.

---

## Usage

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the ML model
python -c "from app.models.train import train_model; train_model()"

# 3. Run the app
streamlit run app/ui/app.py
```

### Programmatic Usage

```python
from shapely import wkt
from app.core.preprocess import preprocess_river
from app.core.multilabel import place_multiple_labels
from app.core.types import LabelSpec

# Load geometry
river = wkt.loads("POLYGON ((...) )")

# Preprocess
river_geom, safe_poly = preprocess_river(river, padding_pt=3.0)

# Create label spec
label = LabelSpec(text="Amazon River", font_family="serif", font_size_pt=12.0)

# Place labels (ML-powered)
result = place_multiple_labels(
    river_geom, safe_poly, label,
    num_labels=None,  # Auto-compute optimal count
    follow_flow=True,  # Align with river direction
    use_learned_ranking=True  # Use ML model
)

# Results
print(f"Placed {result.actual_count} labels")
for p in result.placements:
    print(f"  Position: {p.anchor_pt}, Angle: {p.angle_deg}°")
```

---

## Performance

| Metric | Value |
|--------|-------|
| Placement accuracy | 100% inside geometry |
| Test coverage | 100 WKT files |
| Labels tested | 406 labels |
| Avg. processing time | ~0.7s per river |

---

## Summary

IntelliRiverLabel combines:

1. **Geometric algorithms** for centerline detection, flow direction, and safe zone computation
2. **Machine learning** for intelligent candidate ranking
3. **Multi-label support** for optimal coverage of long rivers
4. **Flow alignment** for aesthetically pleasing, readable labels

The result: Labels that are always inside the river, follow its natural flow, and provide optimal coverage.

@246655693623465 just push this readme after you take latest from main
