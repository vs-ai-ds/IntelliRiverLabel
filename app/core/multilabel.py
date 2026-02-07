# app/core/multilabel.py
"""
Multi-label placement with river flow direction support.
Places multiple labels along a river geometry, ensuring:
1. Labels follow the river flow direction
2. Labels don't overlap with each other
3. Optimal number of labels based on river length

This module integrates with existing placement logic without modifying it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points

from app.core.config import PADDING_PT, SEED, LABEL_BUFFER_EXTRA_PT
from app.core.geometry import (
    bbox_pt_to_polygon,
    oriented_rectangle,
    polygon_bounds,
    sample_points_in_polygon,
)
from app.core.path_b import build_internal_path_polyline, path_length
from app.core.placement import run_placement, _placement_to_collision_geom
from app.core.preprocess import preprocess_river
from app.core.text_metrics import measure_text_pt
from app.core.types import LabelSpec, PlacementResult
from app.core.validate import validate_rect_inside_safe


@dataclass
class FlowDirection:
    """Represents river flow direction."""
    start_pt: Tuple[float, float]
    end_pt: Tuple[float, float]
    angle_deg: float  # Direction angle (0 = right, 90 = up)
    centerline: LineString | None
    
    def reverse(self) -> "FlowDirection":
        """Return reversed flow direction."""
        reversed_angle = (self.angle_deg + 180) % 360
        return FlowDirection(
            start_pt=self.end_pt,
            end_pt=self.start_pt,
            angle_deg=reversed_angle,
            centerline=LineString(list(self.centerline.coords)[::-1]) if self.centerline else None,
        )


@dataclass
class MultiLabelResult:
    """Result of multi-label placement."""
    placements: List[PlacementResult]
    flow_direction: FlowDirection
    optimal_count: int
    actual_count: int
    coverage_ratio: float  # How much of river length is labeled
    warnings: List[str]


def compute_centerline(geom: BaseGeometry, seed: int | None = SEED) -> LineString | None:
    """
    Compute the centerline/skeleton of a river polygon.
    Uses the existing path building logic from Phase B.
    """
    try:
        # Use the existing internal path builder
        path = build_internal_path_polyline(geom, seed)
        if path and not path.is_empty:
            return path
    except Exception:
        pass
    
    # Fallback: approximate centerline using medial axis approach
    try:
        from shapely.geometry import MultiPoint
        
        # Sample points along boundary
        if isinstance(geom, Polygon):
            boundary = geom.exterior
        elif isinstance(geom, MultiPolygon):
            # Use largest polygon
            largest = max(geom.geoms, key=lambda g: g.area)
            boundary = largest.exterior
        else:
            return None
        
        # Sample boundary points
        n_samples = 100
        points = []
        for i in range(n_samples):
            d = boundary.length * i / n_samples
            pt = boundary.interpolate(d)
            points.append((pt.x, pt.y))
        
        # Compute approximate centerline using PCA for proper angle
        minx, miny, maxx, maxy = geom.bounds
        cx = (minx + maxx) / 2
        cy = (miny + maxy) / 2
        
        # Use PCA to get dominant direction
        coords = np.array(points)
        coords_centered = coords - np.mean(coords, axis=0)
        cov = np.cov(coords_centered.T)
        if cov.size > 0:
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx = np.argmax(eigvals)
            v = eigvecs[:, idx]
            
            # Create line along principal axis through centroid
            length = max(maxx - minx, maxy - miny) * 0.8
            start = (cx - v[0] * length / 2, cy - v[1] * length / 2)
            end = (cx + v[0] * length / 2, cy + v[1] * length / 2)
            return LineString([start, end])
        else:
            # Fallback: use bounding box diagonal
            if maxx - minx > maxy - miny:
                return LineString([(minx + 5, cy), (maxx - 5, cy)])
            else:
                return LineString([(cx, miny + 5), (cx, maxy - 5)])
    except Exception:
        return None


def detect_flow_direction(
    geom: BaseGeometry,
    safe_poly: BaseGeometry | None = None,
    seed: int | None = SEED,
) -> FlowDirection:
    """
    Detect the flow direction of a river geometry.
    
    The flow direction is determined by:
    1. Computing the centerline/skeleton
    2. Assuming flow goes from the "wider" end to the "narrower" end
       (or left-to-right / bottom-to-top as default)
    
    Returns FlowDirection with start/end points and angle.
    """
    centerline = compute_centerline(geom, seed)
    
    if centerline is None or centerline.is_empty:
        # Fallback: use bounding box diagonal
        minx, miny, maxx, maxy = geom.bounds
        start = (minx, (miny + maxy) / 2)
        end = (maxx, (miny + maxy) / 2)
        angle = 0.0
        return FlowDirection(start_pt=start, end_pt=end, angle_deg=angle, centerline=None)
    
    coords = list(centerline.coords)
    if len(coords) < 2:
        minx, miny, maxx, maxy = geom.bounds
        return FlowDirection(
            start_pt=(minx, (miny + maxy) / 2),
            end_pt=(maxx, (miny + maxy) / 2),
            angle_deg=0.0,
            centerline=centerline,
        )
    
    start = coords[0]
    end = coords[-1]
    
    # Determine flow direction by comparing widths at start and end
    # Rivers typically flow from narrow to wide (mountain to delta)
    # But we use a heuristic: left-to-right or bottom-to-top as default
    
    # Check if we should reverse (if end is "higher" or more "left")
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    
    # Prefer left-to-right or bottom-to-top
    should_reverse = False
    if abs(dx) > abs(dy):
        # Horizontal dominant - prefer left to right
        if dx < 0:
            should_reverse = True
    else:
        # Vertical dominant - prefer bottom to top
        if dy < 0:
            should_reverse = True
    
    if should_reverse:
        coords = coords[::-1]
        start, end = end, start
        centerline = LineString(coords)
    
    # Compute overall flow angle
    angle_rad = math.atan2(end[1] - start[1], end[0] - start[0])
    angle_deg = math.degrees(angle_rad)
    
    return FlowDirection(
        start_pt=start,
        end_pt=end,
        angle_deg=angle_deg,
        centerline=centerline,
    )


def compute_local_flow_angle(
    centerline: LineString,
    position: Tuple[float, float],
) -> float:
    """
    Compute the local flow angle at a specific position along the centerline.
    This ensures labels follow the river's curvature.
    """
    if centerline is None or centerline.is_empty:
        return 0.0
    
    # Project position onto centerline
    pt = Point(position)
    proj_dist = centerline.project(pt)
    
    # Get tangent direction at this point
    total_len = centerline.length
    if total_len <= 0:
        return 0.0
    
    # Sample two nearby points to get tangent
    delta = min(5.0, total_len * 0.05)  # 5 pt or 5% of length
    d1 = max(0, proj_dist - delta)
    d2 = min(total_len, proj_dist + delta)
    
    p1 = centerline.interpolate(d1)
    p2 = centerline.interpolate(d2)
    
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    # Normalize to [-90, 90] for better label readability
    # Labels should not be upside down
    while angle_deg > 90:
        angle_deg -= 180
    while angle_deg < -90:
        angle_deg += 180
    
    return angle_deg


def estimate_optimal_label_count(
    geom: BaseGeometry,
    label: LabelSpec,
    min_spacing_factor: float = 1.5,  # Reduced for more labels
) -> int:
    """
    Estimate the optimal number of labels for a geometry.
    
    Based on:
    - River length (centerline length)
    - Label width
    - Minimum spacing between labels
    
    Returns recommended number of labels (1 to N).
    """
    w_pt, h_pt = measure_text_pt(label.text, label.font_family, label.font_size_pt)
    if w_pt <= 0:
        return 1
    
    # Get centerline length
    centerline = compute_centerline(geom)
    if centerline is None or centerline.is_empty:
        # Fallback: use bounding box diagonal
        minx, miny, maxx, maxy = geom.bounds
        river_length = math.hypot(maxx - minx, maxy - miny)
    else:
        river_length = path_length(centerline)
    
    if river_length <= 0:
        return 1
    
    # Each label needs: label_width + spacing
    # Spacing = label_width * min_spacing_factor
    space_per_label = w_pt * (1 + min_spacing_factor)
    
    optimal = int(river_length / space_per_label)
    
    # Ensure at least 2 labels for rivers that are long enough
    if river_length > w_pt * 3:
        optimal = max(2, optimal)
    
    # Clamp to reasonable range
    return max(1, min(optimal, 10))


def place_multiple_labels(
    river_geom: BaseGeometry,
    safe_poly: BaseGeometry,
    label: LabelSpec,
    geometry_source: str = "",
    seed: int | None = SEED,
    num_labels: int | None = None,  # None = auto-compute
    min_spacing_factor: float = 1.2,  # Lower for more labels
    follow_flow: bool = True,
    use_learned_ranking: bool = False,
) -> MultiLabelResult:
    """
    Place multiple labels along a river geometry.
    
    Args:
        river_geom: The river polygon
        safe_poly: The safe region for placement (buffered inward)
        label: Label specification
        geometry_source: Source identifier
        seed: Random seed for reproducibility
        num_labels: Number of labels to place (None = auto)
        min_spacing_factor: Minimum spacing between labels as multiple of label width
        follow_flow: If True, align labels with river flow direction
        use_learned_ranking: Use ML model for scoring
        
    Returns:
        MultiLabelResult with all placements and metadata
    """
    warnings: List[str] = []
    
    # Compute flow direction
    flow = detect_flow_direction(river_geom, safe_poly, seed)
    
    # Determine number of labels
    user_specified_count = num_labels is not None
    if num_labels is None:
        optimal_count = estimate_optimal_label_count(river_geom, label, min_spacing_factor)
    else:
        optimal_count = max(1, num_labels)
    
    # For very short rivers, place single label (but respect user's manual choice)
    w_pt, _ = measure_text_pt(label.text, label.font_family, label.font_size_pt)
    centerline = flow.centerline or compute_centerline(river_geom, seed)
    river_length = path_length(centerline) if centerline else 0
    
    # Only fall back to single label if:
    # - User didn't manually specify count AND optimal is 1
    # - OR river is extremely short (less than 1.5x label width)
    force_single = (not user_specified_count and optimal_count == 1) or river_length < w_pt * 1.5
    
    if force_single and optimal_count <= 1:
        # Single label - use existing placement with flow direction
        result = _place_single_label_with_flow(
            river_geom, safe_poly, label, flow,
            geometry_source, seed, use_learned_ranking
        )
        return MultiLabelResult(
            placements=[result] if result else [],
            flow_direction=flow,
            optimal_count=optimal_count,
            actual_count=1 if result else 0,
            coverage_ratio=1.0 if result else 0.0,
            warnings=warnings,
        )
    
    # Multiple labels - place along centerline
    placements: List[PlacementResult] = []
    occupied: BaseGeometry | None = None
    
    w_pt, h_pt = measure_text_pt(label.text, label.font_family, label.font_size_pt)
    
    # Get positions along centerline
    centerline = flow.centerline or compute_centerline(river_geom, seed)
    if centerline is None or centerline.is_empty:
        warnings.append("Could not compute centerline - falling back to single label")
        result = run_placement(
            river_geom, safe_poly, label, geometry_source,
            seed=seed, use_learned_ranking=use_learned_ranking,
        )
        return MultiLabelResult(
            placements=[result],
            flow_direction=flow,
            optimal_count=optimal_count,
            actual_count=1,
            coverage_ratio=1.0 / optimal_count,
            warnings=warnings,
        )
    
    total_length = path_length(centerline)
    label_spacing = total_length / (optimal_count + 1)
    
    for i in range(optimal_count):
        # Position along centerline
        dist = label_spacing * (i + 1)
        pt = centerline.interpolate(dist)
        target_x, target_y = pt.x, pt.y
        
        # Get local flow angle
        if follow_flow:
            local_angle = compute_local_flow_angle(centerline, (target_x, target_y))
        else:
            local_angle = 0.0
        
        # Try to place label at this position
        result = _try_place_at_position(
            river_geom, safe_poly, label,
            target_x, target_y, local_angle,
            geometry_source, seed, occupied,
            use_learned_ranking
        )
        
        if result is not None:
            placements.append(result)
            # Update occupied region
            coll_geom = _placement_to_collision_geom(result)
            if occupied is None:
                occupied = coll_geom
            else:
                occupied = occupied.union(coll_geom)
        else:
            warnings.append(f"Could not place label {i+1} at position ({target_x:.1f}, {target_y:.1f})")
    
    # Calculate coverage
    total_label_length = len(placements) * w_pt
    coverage = total_label_length / total_length if total_length > 0 else 0.0
    
    return MultiLabelResult(
        placements=placements,
        flow_direction=flow,
        optimal_count=optimal_count,
        actual_count=len(placements),
        coverage_ratio=coverage,
        warnings=warnings,
    )


def _place_single_label_with_flow(
    river_geom: BaseGeometry,
    safe_poly: BaseGeometry,
    label: LabelSpec,
    flow: FlowDirection,
    geometry_source: str,
    seed: int | None,
    use_learned_ranking: bool,
) -> PlacementResult | None:
    """
    Place a single label aligned with flow direction.
    Ensures label is INSIDE geometry with proper flow alignment.
    Handles larger font sizes by searching for valid positions.
    """
    # Get text dimensions
    w_pt, h_pt = measure_text_pt(label.text, label.font_family, label.font_size_pt)
    if w_pt <= 0 or h_pt <= 0:
        return None
    
    # Get flow angle at centroid
    if flow.centerline is not None:
        centroid = safe_poly.centroid
        local_angle = compute_local_flow_angle(flow.centerline, (centroid.x, centroid.y))
    else:
        local_angle = flow.angle_deg
    
    def try_placement(x: float, y: float, angle: float) -> Tuple[bool, float, Polygon]:
        """Try placing label at position with angle."""
        rect = oriented_rectangle(x, y, w_pt, h_pt, angle)
        ok, clearance = validate_rect_inside_safe(safe_poly, rect)
        # Also check against original river geometry for extra safety
        if ok:
            ok2, _ = validate_rect_inside_safe(river_geom, rect)
            if not ok2:
                ok = False
        return ok, clearance, rect
    
    # First try: place at safe poly centroid with flow angle
    centroid = safe_poly.centroid
    cx, cy = centroid.x, centroid.y
    
    ok, clearance, rect = try_placement(cx, cy, local_angle)
    if ok:
        return PlacementResult(
            label_text=label.text,
            font_size_pt=label.font_size_pt,
            font_family=label.font_family,
            geometry_source=geometry_source,
            units="pt",
            mode="phase_a_flow_aligned",
            confidence=0.9,
            anchor_pt=(cx, cy),
            angle_deg=local_angle,
            bbox_pt=list(rect.exterior.coords)[:4],
            path_pt=None,
            min_clearance_pt=clearance,
            fit_margin_ratio=clearance / PADDING_PT if PADDING_PT > 0 else 1.0,
            curvature_total_deg=0.0,
            straightness_ratio=1.0,
            warnings=["Placed at centroid with flow alignment"],
        )
    
    # Second try: search along centerline for a position that fits
    if flow.centerline is not None:
        centerline = flow.centerline
        total_len = centerline.length
        
        # Try multiple positions along centerline
        for frac in [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8]:
            pt = centerline.interpolate(frac * total_len)
            test_angle = compute_local_flow_angle(centerline, (pt.x, pt.y))
            
            ok, clearance, rect = try_placement(pt.x, pt.y, test_angle)
            if ok:
                return PlacementResult(
                    label_text=label.text,
                    font_size_pt=label.font_size_pt,
                    font_family=label.font_family,
                    geometry_source=geometry_source,
                    units="pt",
                    mode="phase_a_flow_aligned",
                    confidence=0.85,
                    anchor_pt=(pt.x, pt.y),
                    angle_deg=test_angle,
                    bbox_pt=list(rect.exterior.coords)[:4],
                    path_pt=None,
                    min_clearance_pt=clearance,
                    fit_margin_ratio=clearance / PADDING_PT if PADDING_PT > 0 else 1.0,
                    curvature_total_deg=0.0,
                    straightness_ratio=1.0,
                    warnings=["Placed along centerline with flow alignment"],
                )
    
    # Third try: sample points inside safe_poly
    sample_pts = sample_points_in_polygon(safe_poly, 20, seed)
    for px, py in sample_pts:
        if flow.centerline is not None:
            test_angle = compute_local_flow_angle(flow.centerline, (px, py))
        else:
            test_angle = local_angle
        
        ok, clearance, rect = try_placement(px, py, test_angle)
        if ok:
            return PlacementResult(
                label_text=label.text,
                font_size_pt=label.font_size_pt,
                font_family=label.font_family,
                geometry_source=geometry_source,
                units="pt",
                mode="phase_a_flow_aligned",
                confidence=0.75,
                anchor_pt=(px, py),
                angle_deg=test_angle,
                bbox_pt=list(rect.exterior.coords)[:4],
                path_pt=None,
                min_clearance_pt=clearance,
                fit_margin_ratio=clearance / PADDING_PT if PADDING_PT > 0 else 1.0,
                curvature_total_deg=0.0,
                straightness_ratio=1.0,
                warnings=["Placed at sampled interior point"],
            )
    
    # Fourth try: use standard placement and adjust angle
    result = run_placement(
        river_geom, safe_poly, label, geometry_source,
        seed=seed, use_learned_ranking=use_learned_ranking,
    )
    
    if result.mode == "external_fallback":
        return result
    
    # Adjust angle to match local flow direction at placement position
    if flow.centerline is not None:
        local_angle = compute_local_flow_angle(flow.centerline, result.anchor_pt)
    
    # Recompute bbox with flow angle
    cx, cy = result.anchor_pt
    ok, clearance, rect = try_placement(cx, cy, local_angle)
    
    if ok:
        return PlacementResult(
            label_text=result.label_text,
            font_size_pt=result.font_size_pt,
            font_family=result.font_family,
            geometry_source=result.geometry_source,
            units=result.units,
            mode="phase_a_flow_aligned",
            confidence=result.confidence,
            anchor_pt=result.anchor_pt,
            angle_deg=local_angle,
            bbox_pt=list(rect.exterior.coords)[:4],
            path_pt=result.path_pt,
            min_clearance_pt=clearance,
            fit_margin_ratio=result.fit_margin_ratio,
            curvature_total_deg=result.curvature_total_deg,
            straightness_ratio=result.straightness_ratio,
            warnings=result.warnings + ["Angle adjusted to match flow direction"],
        )
    
    # Flow-aligned doesn't fit, return original
    return result


def _try_place_at_position(
    river_geom: BaseGeometry,
    safe_poly: BaseGeometry,
    label: LabelSpec,
    target_x: float,
    target_y: float,
    target_angle: float,
    geometry_source: str,
    seed: int | None,
    occupied: BaseGeometry | None,
    use_learned_ranking: bool,
) -> PlacementResult | None:
    """
    Try to place a label at a specific target position.
    Falls back to nearby positions if target doesn't work.
    Ensures label is INSIDE the geometry with proper angle.
    """
    w_pt, h_pt = measure_text_pt(label.text, label.font_family, label.font_size_pt)
    if w_pt <= 0 or h_pt <= 0:
        return None
    
    def try_position(x: float, y: float, angle: float) -> Tuple[bool, float, Polygon]:
        """Check if position works. Returns (ok, clearance, rect)."""
        # Check if point is inside or very close to safe poly
        pt = Point(x, y)
        if not (safe_poly.contains(pt) or safe_poly.distance(pt) < 1.0):
            return False, 0.0, Polygon()
        
        rect = oriented_rectangle(x, y, w_pt, h_pt, angle)
        ok, clearance = validate_rect_inside_safe(safe_poly, rect)
        return ok, clearance, rect
    
    def check_collision(rect: Polygon) -> bool:
        """Check if rect collides with occupied. Returns True if OK (no collision)."""
        if occupied is None or occupied.is_empty:
            return True
        rect_poly = bbox_pt_to_polygon(list(rect.exterior.coords)[:4])
        inter = rect_poly.intersection(occupied)
        return inter.is_empty or inter.area <= 5.0
    
    # Try target position first with given angle
    ok, clearance, rect = try_position(target_x, target_y, target_angle)
    if ok and check_collision(rect):
        return PlacementResult(
            label_text=label.text,
            font_size_pt=label.font_size_pt,
            font_family=label.font_family,
            geometry_source=geometry_source,
            units="pt",
            mode="phase_a_flow_aligned",
            confidence=0.9,
            anchor_pt=(target_x, target_y),
            angle_deg=target_angle,
            bbox_pt=list(rect.exterior.coords)[:4],
            path_pt=None,
            min_clearance_pt=clearance,
            fit_margin_ratio=clearance / PADDING_PT if PADDING_PT > 0 else 1.0,
            curvature_total_deg=0.0,
            straightness_ratio=1.0,
            warnings=[],
        )
    
    # Try different angles at target position
    for angle_adj in [0, 10, -10, 20, -20, 30, -30, 45, -45]:
        test_angle = target_angle + angle_adj
        ok, clearance, rect = try_position(target_x, target_y, test_angle)
        if ok and check_collision(rect):
            return PlacementResult(
                label_text=label.text,
                font_size_pt=label.font_size_pt,
                font_family=label.font_family,
                geometry_source=geometry_source,
                units="pt",
                mode="phase_a_flow_aligned",
                confidence=0.85,
                anchor_pt=(target_x, target_y),
                angle_deg=test_angle,
                bbox_pt=list(rect.exterior.coords)[:4],
                path_pt=None,
                min_clearance_pt=clearance,
                fit_margin_ratio=clearance / PADDING_PT if PADDING_PT > 0 else 1.0,
                curvature_total_deg=0.0,
                straightness_ratio=1.0,
                warnings=["Angle adjusted for fit"],
            )
    
    # Try nearby positions with original angle (spiral search)
    search_radius = max(w_pt, h_pt) * 0.5
    for r in [search_radius * 0.3, search_radius * 0.6, search_radius, search_radius * 1.5, search_radius * 2.0]:
        for angle_offset in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
            rad = math.radians(angle_offset)
            test_x = target_x + r * math.cos(rad)
            test_y = target_y + r * math.sin(rad)
            
            # Try with original angle first
            ok, clearance, rect = try_position(test_x, test_y, target_angle)
            if ok and check_collision(rect):
                return PlacementResult(
                    label_text=label.text,
                    font_size_pt=label.font_size_pt,
                    font_family=label.font_family,
                    geometry_source=geometry_source,
                    units="pt",
                    mode="phase_a_flow_aligned",
                    confidence=0.7,
                    anchor_pt=(test_x, test_y),
                    angle_deg=target_angle,
                    bbox_pt=list(rect.exterior.coords)[:4],
                    path_pt=None,
                    min_clearance_pt=clearance,
                    fit_margin_ratio=clearance / PADDING_PT if PADDING_PT > 0 else 1.0,
                    curvature_total_deg=0.0,
                    straightness_ratio=1.0,
                    warnings=["Position adjusted from target"],
                )
            
            # Try with adjusted angles too
            for angle_adj in [15, -15, 30, -30]:
                test_angle = target_angle + angle_adj
                ok, clearance, rect = try_position(test_x, test_y, test_angle)
                if ok and check_collision(rect):
                    return PlacementResult(
                        label_text=label.text,
                        font_size_pt=label.font_size_pt,
                        font_family=label.font_family,
                        geometry_source=geometry_source,
                        units="pt",
                        mode="phase_a_flow_aligned",
                        confidence=0.65,
                        anchor_pt=(test_x, test_y),
                        angle_deg=test_angle,
                        bbox_pt=list(rect.exterior.coords)[:4],
                        path_pt=None,
                        min_clearance_pt=clearance,
                        fit_margin_ratio=clearance / PADDING_PT if PADDING_PT > 0 else 1.0,
                        curvature_total_deg=0.0,
                        straightness_ratio=1.0,
                        warnings=["Position and angle adjusted"],
                    )
    
    # Last resort: sample random points in safe_poly
    try:
        sample_pts = sample_points_in_polygon(safe_poly, 15, seed)
        for px, py in sample_pts:
            for angle_adj in [0, 15, -15, 30, -30]:
                test_angle = target_angle + angle_adj
                ok, clearance, rect = try_position(px, py, test_angle)
                if ok and check_collision(rect):
                    return PlacementResult(
                        label_text=label.text,
                        font_size_pt=label.font_size_pt,
                        font_family=label.font_family,
                        geometry_source=geometry_source,
                        units="pt",
                        mode="phase_a_flow_aligned",
                        confidence=0.5,
                        anchor_pt=(px, py),
                        angle_deg=test_angle,
                        bbox_pt=list(rect.exterior.coords)[:4],
                        path_pt=None,
                        min_clearance_pt=clearance,
                        fit_margin_ratio=clearance / PADDING_PT if PADDING_PT > 0 else 1.0,
                        curvature_total_deg=0.0,
                        straightness_ratio=1.0,
                        warnings=["Placed at sampled position"],
                    )
    except Exception:
        pass
    
    return None


def correct_existing_placement(
    river_geom: BaseGeometry,
    safe_poly: BaseGeometry,
    existing_result: PlacementResult,
    label: LabelSpec,
) -> PlacementResult:
    """
    Correct an existing placement to follow flow direction and be inside geometry.
    
    This function takes an existing placement and:
    1. Ensures it's inside the geometry
    2. Adjusts angle to follow flow direction
    
    Returns corrected PlacementResult.
    """
    # Detect flow direction
    flow = detect_flow_direction(river_geom, safe_poly)
    
    # Get local flow angle at current position
    if flow.centerline is not None:
        local_angle = compute_local_flow_angle(flow.centerline, existing_result.anchor_pt)
    else:
        local_angle = flow.angle_deg
    
    cx, cy = existing_result.anchor_pt
    w_pt, h_pt = measure_text_pt(label.text, label.font_family, label.font_size_pt)
    
    # Check if current position with flow angle works
    rect = oriented_rectangle(cx, cy, w_pt, h_pt, local_angle)
    ok, clearance = validate_rect_inside_safe(safe_poly, rect)
    
    if ok:
        return PlacementResult(
            label_text=existing_result.label_text,
            font_size_pt=existing_result.font_size_pt,
            font_family=existing_result.font_family,
            geometry_source=existing_result.geometry_source,
            units=existing_result.units,
            mode="phase_a_flow_corrected",
            confidence=existing_result.confidence,
            anchor_pt=existing_result.anchor_pt,
            angle_deg=local_angle,
            bbox_pt=list(rect.exterior.coords)[:4],
            path_pt=existing_result.path_pt,
            min_clearance_pt=clearance,
            fit_margin_ratio=existing_result.fit_margin_ratio,
            curvature_total_deg=existing_result.curvature_total_deg,
            straightness_ratio=existing_result.straightness_ratio,
            warnings=existing_result.warnings + ["Corrected to follow flow direction"],
        )
    
    # Need to find a new position
    # Project current position onto centerline and find nearby valid spot
    if flow.centerline is not None:
        pt = Point(cx, cy)
        proj_dist = flow.centerline.project(pt)
        proj_pt = flow.centerline.interpolate(proj_dist)
        
        result = _try_place_at_position(
            river_geom, safe_poly, label,
            proj_pt.x, proj_pt.y, local_angle,
            existing_result.geometry_source, None, None, False
        )
        
        if result is not None:
            result.warnings = existing_result.warnings + ["Position corrected to follow flow and fit inside"]
            return result
    
    # Fallback: return original with warning
    return PlacementResult(
        label_text=existing_result.label_text,
        font_size_pt=existing_result.font_size_pt,
        font_family=existing_result.font_family,
        geometry_source=existing_result.geometry_source,
        units=existing_result.units,
        mode=existing_result.mode,
        confidence=existing_result.confidence * 0.5,
        anchor_pt=existing_result.anchor_pt,
        angle_deg=existing_result.angle_deg,
        bbox_pt=existing_result.bbox_pt,
        path_pt=existing_result.path_pt,
        min_clearance_pt=existing_result.min_clearance_pt,
        fit_margin_ratio=existing_result.fit_margin_ratio,
        curvature_total_deg=existing_result.curvature_total_deg,
        straightness_ratio=existing_result.straightness_ratio,
        warnings=existing_result.warnings + ["WARNING: Could not correct placement to follow flow"],
    )
