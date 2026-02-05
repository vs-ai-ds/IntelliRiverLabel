# app/core/types.py
"""
Dataclasses for label spec, placement result, and candidates.
Schema aligns with docs/PLACEMENT_SCHEMA.md.
See: docs/PROJECT_SPEC.md, docs/PLACEMENT_SCHEMA.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


PlacementMode = Literal["phase_a_straight", "phase_b_curved", "external_fallback"]


@dataclass(frozen=True)
class LabelSpec:
    """Label text and typography. See docs/PLACEMENT_SCHEMA.md (label)."""
    text: str
    font_family: str
    font_size_pt: float


@dataclass
class CandidatePoint:
    """A candidate placement point with clearance and score. See docs/ALGORITHM.md A2."""
    x: float
    y: float
    clearance: float
    base_score: float
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class PlacementResult:
    """
    Full placement output matching docs/PLACEMENT_SCHEMA.md.
    Serializes to placement.json.
    """
    # label
    label_text: str
    font_size_pt: float
    font_family: str

    # input
    geometry_source: str

    # result
    mode: PlacementMode
    confidence: float
    anchor_pt: tuple[float, float]  # (x, y)
    angle_deg: float
    bbox_pt: list[tuple[float, float]]  # 4 corners

    # metrics
    min_clearance_pt: float
    fit_margin_ratio: float
    curvature_total_deg: float
    straightness_ratio: float

    # optional / defaults (must follow required fields in dataclass)
    units: str = "pt"
    path_pt: list[tuple[float, float]] | None = None  # only for phase_b_curved
    warnings: list[str] = field(default_factory=list)
