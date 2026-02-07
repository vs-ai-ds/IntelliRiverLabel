# Placement JSON Schema

Schema **version**: `1.0` (see `schema_version` in placement.json). Increment when making breaking changes.

## File: placement.json

{
  "schema_version": "1.0",
  "label": {
    "text": "ELBE",
    "font_size_pt": 12,
    "font_family": "DejaVu Sans"
  },
  "input": {
    "geometry_source": "../Problem/river.wkt",
    "units": "pt"
  },
  "result": {
    "mode": "phase_a_straight" | "phase_b_curved" | "external_fallback",
    "confidence": 0.0,
    "anchor_pt": {"x": 0.0, "y": 0.0},
    "angle_deg": 0.0,
    "bbox_pt": [{"x":0,"y":0},{"x":0,"y":0},{"x":0,"y":0},{"x":0,"y":0}],
    "path_pt": [{"x":0,"y":0}]  // present only for phase_b_curved
  },
  "metrics": {
    "min_clearance_pt": 0.0,
    "fit_margin_ratio": 0.0,
    "curvature_total_deg": 0.0,
    "straightness_ratio": 0.0
  },
  "warnings": []
}