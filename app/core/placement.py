# app/core/placement.py
"""
Full Phase A placement pipeline. External fallback only if no internal placement.
Returns PlacementResult with metrics and warnings. See: docs/ALGORITHM.md, docs/PLACEMENT_SCHEMA.md.
"""

from __future__ import annotations

from shapely.geometry.base import BaseGeometry

from app.core.candidates_a import angle_candidates_deg, generate_candidate_points
from app.core.config import PADDING_PT, SEED
from app.core.geometry import polygon_bounds, oriented_rectangle
from app.core.scoring import (
    centering_score,
    fit_margin_ratio,
    score_phase_a,
)
from app.core.text_metrics import measure_text_pt
from app.core.types import LabelSpec, PlacementResult
from app.core.validate import validate_rect_inside_safe


def _run_phase_a(
    safe_poly: BaseGeometry,
    river_geom: BaseGeometry,
    label: LabelSpec,
    geometry_source: str,
    seed: int | None = SEED,
) -> tuple[PlacementResult | None, list[str]]:
    """
    Try internal placement. Returns (PlacementResult, warnings) or (None, warnings).
    """
    warnings: list[str] = []
    w_pt, h_pt = measure_text_pt(label.text, label.font_family, label.font_size_pt)
    if w_pt <= 0 or h_pt <= 0:
        warnings.append("Label has zero size from font metrics.")
        return None, warnings

    candidates = generate_candidate_points(safe_poly, seed=seed)
    if not candidates:
        warnings.append("No candidate points in safe polygon.")
        return None, warnings

    angles = angle_candidates_deg(river_geom)
    if not angles:
        angles = [0.0]

    bounds = polygon_bounds(safe_poly)
    best: tuple[float, float, float, float, float, float] | None = None
    best_score = float("-inf")

    for cp in candidates:
        for angle_deg in angles:
            rect = oriented_rectangle(cp.x, cp.y, w_pt, h_pt, angle_deg)
            ok, min_clearance_pt = validate_rect_inside_safe(safe_poly, rect)
            if not ok:
                continue
            fit = fit_margin_ratio(min_clearance_pt, PADDING_PT)
            centering = centering_score(cp.x, cp.y, bounds)
            score = score_phase_a(cp.clearance, fit, centering, angle_deg)
            if score > best_score:
                best_score = score
                best = (cp.x, cp.y, angle_deg, min_clearance_pt, fit, centering)

    if best is None:
        warnings.append("No feasible internal placement found.")
        return None, warnings

    cx, cy, angle_deg, min_clearance_pt, fit_margin, _ = best
    rect = oriented_rectangle(cx, cy, w_pt, h_pt, angle_deg)
    bbox_pt = list(rect.exterior.coords)[:4]

    result = PlacementResult(
        label_text=label.text,
        font_size_pt=label.font_size_pt,
        font_family=label.font_family,
        geometry_source=geometry_source,
        units="pt",
        mode="phase_a_straight",
        confidence=min(1.0, 0.5 + fit_margin * 0.5),
        anchor_pt=(cx, cy),
        angle_deg=angle_deg,
        bbox_pt=bbox_pt,
        path_pt=None,
        min_clearance_pt=min_clearance_pt,
        fit_margin_ratio=fit_margin,
        curvature_total_deg=0.0,
        straightness_ratio=1.0,
        warnings=warnings,
    )
    return result, warnings


def _external_fallback(
    river_geom: BaseGeometry,
    label: LabelSpec,
    geometry_source: str,
    warnings: list[str],
) -> PlacementResult:
    """Place label at centroid when internal placement fails. See: docs/ALGORITHM.md."""
    w_pt, h_pt = measure_text_pt(label.text, label.font_family, label.font_size_pt)
    if w_pt <= 0:
        w_pt = 10.0
    if h_pt <= 0:
        h_pt = 10.0
    c = river_geom.centroid
    cx, cy = float(c.x), float(c.y)
    rect = oriented_rectangle(cx, cy, w_pt, h_pt, 0.0)
    bbox_pt = list(rect.exterior.coords)[:4]
    return PlacementResult(
        label_text=label.text,
        font_size_pt=label.font_size_pt,
        font_family=label.font_family,
        geometry_source=geometry_source,
        units="pt",
        mode="external_fallback",
        confidence=0.0,
        anchor_pt=(cx, cy),
        angle_deg=0.0,
        bbox_pt=bbox_pt,
        path_pt=None,
        min_clearance_pt=0.0,
        fit_margin_ratio=0.0,
        curvature_total_deg=0.0,
        straightness_ratio=1.0,
        warnings=warnings + ["External fallback: label placed at centroid."],
    )


def run_placement_phase_a(
    river_geom: BaseGeometry,
    safe_poly: BaseGeometry,
    label: LabelSpec,
    geometry_source: str,
    seed: int | None = SEED,
) -> PlacementResult:
    """
    Full Phase A pipeline. Tries internal placement; if none feasible, external fallback.
    See: docs/ALGORITHM.md, docs/PLACEMENT_SCHEMA.md.
    """
    result, warnings = _run_phase_a(safe_poly, river_geom, label, geometry_source, seed=seed)
    if result is not None:
        return result
    return _external_fallback(river_geom, label, geometry_source, warnings)
