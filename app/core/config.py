# Repo//app/core/config.py
"""
Central configuration for river label placement.
All tunable values live here; no magic numbers in other modules.
See: docs/PROJECT_SPEC.md, docs/ARCHITECTURE.md, docs/ALGORITHM.md.
"""

from __future__ import annotations

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
