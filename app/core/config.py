# app/core/config.py
"""
Central configuration for river label placement.
All tunable values live here; no magic numbers in other modules.
See: docs/PROJECT_SPEC.md, docs/ARCHITECTURE.md, docs/ALGORITHM.md.
"""

from __future__ import annotations
import os

# ----- Paths (repo-relative) -----
DEFAULT_GEOMETRY_PATH: str = "docs/assets/problem/Problem_1_river.wkt"
REPORTS_DIR: str = "reports"

# ----- Padding and safe polygon -----
PADDING_PT: float = 4.0
"""Inward buffer distance (pt) for safe polygon. See ALGORITHM A1."""

MIN_BUFFER_PT: float = 0.5
"""Minimum buffer to try when safe polygon would be empty. See ALGORITHM A1."""

MIN_PADDING_PT: float = 0.5
"""Minimum padding when reducing after empty safe polygon. See ALGORITHM A1."""

# ----- Sampling (Phase A) -----
N_SAMPLE_POINTS: int = 500
"""Number of interior points to sample. See ALGORITHM A2."""

K_TOP_CLEARANCE: int = 50
"""Keep top K candidate points by clearance. See ALGORITHM A2."""

# ----- Angle candidates -----
ANGLE_OFFSETS_DEG: tuple[float, ...] = (0.0, 15.0, 30.0)
"""Offsets in degrees from PCA base angle: base, base ± 15°, base ± 30°. See ALGORITHM A3."""

# ----- Feasibility -----
CONTAINMENT_TOLERANCE_PT: float = 1e-6
"""Tolerance (pt) for rectangle-in-polygon containment. See ALGORITHM A4."""

# ----- Rendering -----
RENDER_WIDTH_PX: int = 800
RENDER_HEIGHT_PX: int = 600

# ----- Scoring (Phase A heuristic). See docs/ALGORITHM.md A5 -----
SCORE_WEIGHT_CLEARANCE: float = 1.0
SCORE_WEIGHT_FIT_MARGIN: float = 1.0
SCORE_WEIGHT_CENTERING: float = 0.5
SCORE_WEIGHT_ANGLE_PENALTY: float = 0.1

# ----- Determinism -----
SEED: int | None = 42
"""Random seed for sampling; None for non-deterministic."""

# ----- Default font -----
DEFAULT_FONT_FAMILY: str = "DejaVu Sans"

# ----- Phase B curved path -----
CURVE_FIT_MARGIN: float = 1.10
"""Path length >= label_width_pt * CURVE_FIT_MARGIN."""

CURVE_EXTRA_CLEARANCE_PT: float = 2.0
"""Extra clearance required along path (on top of padding_pt)."""

CURVE_MAX_POINTS: int = 200
"""Max points in path_pt for placement / SVG."""

PATH_SMOOTHING_WINDOW: int = 7
"""Window size for moving average smoothing of internal path."""

PATH_SAMPLE_STEP_PT: float = 5.0
"""Step (pt) when sampling along path for clearance."""

PATH_SLICES: int = 120
"""Number of cross-section slices along dominant axis for centerline (deterministic)."""

PATH_MIN_POINTS: int = 10
"""Minimum centerline points to return a valid path."""

PATH_WINDOW_STEP_PT: float = 8.0
"""Step (pt) when sliding windows along path for Phase B window selection."""

# ----- Learned ranking (optional) -----
ENABLE_LEARNED_RANKING: bool = True
"""Use trained model to blend with heuristic score when True. Disabled by default for testing multilabel."""

LEARNED_BLEND_ALPHA: float = 0.6
"""Blend: final = alpha * heuristic + (1 - alpha) * model_score. Ignored if model missing."""

# ----- Multi-label collision avoidance -----
COLLISION_WEIGHT: float = 5.0
"""Score penalty weight for overlap with occupied geometry (multi-label)."""

COLLISION_MAX_AREA: float = 0.5
"""Max allowed intersection area (pt²) with occupied; above this candidate is rejected."""

LABEL_BUFFER_EXTRA_PT: float = 1.5
"""Extra buffer (pt) for label collision geometry (bbox or path buffer)."""

# ----- Zoom buckets -----
ZOOM_BUCKETS_DEFAULT: tuple[int, ...] = (10, 12, 14)
"""Default zoom bucket indices for consistent placement across zoom levels."""

ZOOM_FONT_SCALE_FACTOR: float = 0.15
"""Per-bucket font scale: font_pt = base_font * (1.0 + ZOOM_FONT_SCALE_FACTOR * bucket_index)."""

ZOOM_PADDING_SCALE_FACTOR: float = 0.1
"""Per-bucket padding scale: padding_pt = base_padding * (1.0 + ZOOM_PADDING_SCALE_FACTOR * bucket_index)."""

# ----- Debug flags -----
PHASE_B_DEBUG: bool = os.environ.get("PHASE_B_DEBUG", "").lower() in ("1", "true", "yes")
"""Enable Phase B debug output. Set env PHASE_B_DEBUG=1 to enable."""
