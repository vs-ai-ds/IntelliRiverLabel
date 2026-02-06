# app/core/placement.py
"""
Full Phase A placement pipeline. External fallback only if no internal placement.
Returns PlacementResult with metrics and warnings. See: docs/ALGORITHM.md, docs/PLACEMENT_SCHEMA.md.
"""

from __future__ import annotations

from shapely.geometry.base import BaseGeometry

from app.core.candidates_a import angle_candidates_deg, generate_candidate_points
from app.core.config import (
    COLLISION_MAX_AREA,
    COLLISION_WEIGHT,
    LABEL_BUFFER_EXTRA_PT,
    PADDING_PT,
    SEED,
)
from app.core.geometry import bbox_pt_to_polygon, polygon_bounds, oriented_rectangle
from app.core.phase_b import try_phase_b_curved
from app.core.scoring import (
    centering_score,
    fit_margin_ratio,
    score_blended,
    score_phase_a,
)
from app.core.text_metrics import measure_text_pt
from app.core.types import LabelSpec, PlacementResult
from app.core.validate import validate_rect_inside_safe


def _placement_to_collision_geom(result: PlacementResult, buffer_pt: float = LABEL_BUFFER_EXTRA_PT) -> BaseGeometry:
    """Build collision geometry for a placement: Phase A = buffered bbox; Phase B = buffered path or bbox."""
    from shapely.geometry import LineString, Polygon

    if result.path_pt and len(result.path_pt) >= 2:
        line = LineString(result.path_pt)
        return line.buffer(buffer_pt + result.font_size_pt * 0.5) if not line.is_empty else Polygon()
    return bbox_pt_to_polygon(result.bbox_pt, buffer_pt=buffer_pt)


def _run_phase_a(
    safe_poly: BaseGeometry,
    river_geom: BaseGeometry,
    label: LabelSpec,
    geometry_source: str,
    seed: int | None = SEED,
    use_learned_ranking: bool = False,
    occupied: BaseGeometry | None = None,
    collision_weight: float = COLLISION_WEIGHT,
    collision_max_area: float = COLLISION_MAX_AREA,
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
            collision_penalty = 0.0
            if occupied is not None and not occupied.is_empty:
                rect_poly = bbox_pt_to_polygon(list(rect.exterior.coords)[:4], buffer_pt=0)
                inter = rect_poly.intersection(occupied)
                if not inter.is_empty and inter.area > collision_max_area:
                    continue
                if not inter.is_empty and inter.area > 0:
                    collision_penalty = collision_weight * (inter.area / max(rect_poly.area, 1e-6))
            fit = fit_margin_ratio(min_clearance_pt, PADDING_PT)
            centering = centering_score(cp.x, cp.y, bounds)
            heuristic_s = score_phase_a(cp.clearance, fit, centering, angle_deg) - collision_penalty
            if use_learned_ranking:
                from app.models.features import candidate_to_features
                features = candidate_to_features(
                    cp.x, cp.y, cp.clearance, angle_deg,
                    min_clearance_pt, bounds, PADDING_PT,
                )
                score = score_blended(heuristic_s + collision_penalty, features, use_model=use_learned_ranking) - collision_penalty
            else:
                score = heuristic_s
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
    use_learned_ranking: bool = False,
    occupied: BaseGeometry | None = None,
    collision_weight: float = COLLISION_WEIGHT,
    collision_max_area: float = COLLISION_MAX_AREA,
) -> PlacementResult:
    """
    Full Phase A pipeline. Tries internal placement; if none feasible, external fallback.
    When occupied is set, candidates overlapping occupied are penalized or rejected.
    """
    result, warnings = _run_phase_a(
        safe_poly, river_geom, label, geometry_source,
        seed=seed, use_learned_ranking=use_learned_ranking,
        occupied=occupied,
        collision_weight=collision_weight,
        collision_max_area=collision_max_area,
    )
    if result is not None:
        return result
    return _external_fallback(river_geom, label, geometry_source, warnings)


def run_placement(
    river_geom: BaseGeometry,
    safe_poly: BaseGeometry,
    label: LabelSpec,
    geometry_source: str,
    seed: int | None = SEED,
    allow_phase_b: bool = False,
    padding_pt: float = PADDING_PT,
    use_learned_ranking: bool = False,
    occupied: BaseGeometry | None = None,
    collision_weight: float = COLLISION_WEIGHT,
    collision_max_area: float = COLLISION_MAX_AREA,
) -> PlacementResult:
    """
    Run placement with optional Phase B. When occupied is set, avoids/penalizes overlap (multi-label).
    Phase B result is rejected if it collides with occupied.
    """
    if allow_phase_b:
        phase_b_result, phase_b_reason = try_phase_b_curved(
            river_geom,
            safe_poly,
            label,
            padding_pt,
            seed,
            geometry_source=geometry_source,
        )
        if phase_b_result is not None and occupied is not None and not occupied.is_empty:
            coll_geom = _placement_to_collision_geom(phase_b_result)
            inter = coll_geom.intersection(occupied)
            if not inter.is_empty and inter.area > collision_max_area:
                phase_b_result = None
                phase_b_reason = "collides_with_existing"
        if phase_b_result is not None:
            return phase_b_result
    result = run_placement_phase_a(
        river_geom, safe_poly, label, geometry_source,
        seed=seed, use_learned_ranking=use_learned_ranking,
        occupied=occupied,
        collision_weight=collision_weight,
        collision_max_area=collision_max_area,
    )
    if allow_phase_b:
        reason_msg = f"Phase B attempted but failed: {phase_b_reason}"
        return PlacementResult(
            label_text=result.label_text,
            font_size_pt=result.font_size_pt,
            font_family=result.font_family,
            geometry_source=result.geometry_source,
            units=result.units,
            mode=result.mode,
            confidence=result.confidence,
            anchor_pt=result.anchor_pt,
            angle_deg=result.angle_deg,
            bbox_pt=result.bbox_pt,
            path_pt=result.path_pt,
            min_clearance_pt=result.min_clearance_pt,
            fit_margin_ratio=result.fit_margin_ratio,
            curvature_total_deg=result.curvature_total_deg,
            straightness_ratio=result.straightness_ratio,
            warnings=result.warnings + [reason_msg],
        )
    return result
