# app/core/phase_b.py
"""
Phase B curved placement: internal path, window selection, validation.
Returns PlacementResult or None; caller falls back to Phase A. See: docs/ALGORITHM.md.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable

from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

# Ensure warnings/errors are visible (add handler if none exists)
if not logger.handlers and not logging.root.handlers:
    import sys
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.WARNING)
    formatter = logging.Formatter('[Phase B] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

from app.core.candidates_b import propose_path_window
from app.core.config import (
    CURVE_EXTRA_CLEARANCE_PT,
    CURVE_FIT_MARGIN,
    CURVE_MAX_POINTS,
    PATH_SAMPLE_STEP_PT,
    PHASE_B_DEBUG,
)

def _phase_b_print(*args: object, **kwargs: object) -> None:
    """Print only when PHASE_B_DEBUG is True (set env PHASE_B_DEBUG=1)."""
    if PHASE_B_DEBUG:
        print(*args, **kwargs)
from app.core.path_b import (
    _snap_coords_inside,
    build_internal_path_polyline,
    min_clearance_along_path,
    path_length,
    sample_along_path,
)
from app.core.text_metrics import measure_text_pt
from app.core.types import LabelSpec, PlacementResult


def _curvature_total_deg(path: LineString) -> float:
    """Sum of absolute angle changes along path (degrees)."""
    if path is None or path.is_empty:
        return 0.0
    coords = list(path.coords)
    if len(coords) < 3:
        return 0.0
    total = 0.0
    for i in range(1, len(coords) - 1):
        ax = coords[i][0] - coords[i - 1][0]
        ay = coords[i][1] - coords[i - 1][1]
        bx = coords[i + 1][0] - coords[i][0]
        by = coords[i + 1][1] - coords[i][1]
        a_rad = math.atan2(ay, ax)
        b_rad = math.atan2(by, bx)
        delta = math.degrees(abs((b_rad - a_rad + math.pi) % (2 * math.pi) - math.pi))
        total += delta
    return float(total)


def _straightness_ratio(path: LineString) -> float:
    """Chord length / path length; 1 = straight."""
    if path is None or path.is_empty:
        return 1.0
    length = path_length(path)
    if length <= 0:
        return 1.0
    coords = list(path.coords)
    if len(coords) < 2:
        return 1.0
    chord = math.hypot(coords[-1][0] - coords[0][0], coords[-1][1] - coords[0][1])
    if chord <= 0:
        return 1.0
    return float(min(1.0, chord / length))


def _downsample_path(coords: list[tuple[float, float]], max_pts: int) -> list[tuple[float, float]]:
    """Downsample to at most max_pts points, preserving endpoints."""
    if len(coords) <= max_pts:
        return list(coords)
    step = (len(coords) - 1) / (max_pts - 1)
    out = []
    for i in range(max_pts):
        out.append(coords[int(round(i * step))])
    # ensure last exactly last
    out[-1] = coords[-1]
    out[0] = coords[0]
    return out


def _all_points_inside(poly: BaseGeometry, pts: Iterable[Point]) -> bool:
    """Strictly require points on window to be inside or covered by safe polygon."""
    if poly is None or poly.is_empty:
        return False
    try:
        for p in pts:
            if p is None or p.is_empty:
                return False
            # covers allows boundary; contains is strict interior
            if not (poly.covers(p) or poly.contains(p)):
                return False
        return True
    except Exception:
        return False


def try_phase_b_curved(
    polygon: BaseGeometry,
    safe_poly: BaseGeometry,
    label_spec: LabelSpec,
    padding_pt: float,
    seed: int | None,
    geometry_source: str = "",
) -> tuple[PlacementResult | None, str]:
    """
    Attempt Phase B curved placement. Returns (PlacementResult, "") on success, (None, reason) on failure.
    Reason is a short string explaining why Phase B failed (e.g., "no internal path", "insufficient clearance: min=5.0, required=6.0").
    """
    try:
        # Build internal path from SAFE polygon to begin with.
        path = build_internal_path_polyline(safe_poly, seed)
        if path is None or path.is_empty:
            return None, "no internal path"

        # CRITICAL: Clip path to safe polygon to ensure all segments are inside
        # (path might extend beyond safe_poly even after snapping)
        try:
            path_clipped = path.intersection(safe_poly)
            if path_clipped.is_empty:
                _phase_b_print(f"[Phase B DEBUG] Path intersection with safe_poly is empty", flush=True)
                return None, "no internal path"
            from shapely.geometry import MultiLineString, GeometryCollection
            if isinstance(path_clipped, LineString):
                path = path_clipped
            elif isinstance(path_clipped, MultiLineString) and path_clipped.geoms:
                # Take longest segment
                path = max(path_clipped.geoms, key=lambda x: x.length)
            elif isinstance(path_clipped, GeometryCollection):
                lines = [g for g in path_clipped.geoms if isinstance(g, LineString) and not g.is_empty]
                if not lines:
                    return None, "no internal path"
                path = max(lines, key=lambda x: x.length)
            else:
                return None, "no internal path"
            _phase_b_print(f"[Phase B DEBUG] Path clipped to safe_poly, new length: {path_length(path):.2f}", flush=True)
        except Exception as e:
            _phase_b_print(f"[Phase B DEBUG] Path clipping failed: {e}, using original path", flush=True)
            # Continue with original path (might have issues, but try anyway)

        w_pt, _ = measure_text_pt(label_spec.text, label_spec.font_family, label_spec.font_size_pt)
        if w_pt <= 0:
            return None, "label width invalid"

        required_len = float(w_pt * CURVE_FIT_MARGIN)
        if required_len <= 0:
            return None, "label width invalid"

        # Pick best window (candidate) from the centerline (now guaranteed inside safe_poly)
        path_len = path_length(path)
        _phase_b_print(f"[Phase B DEBUG] Path length: {path_len:.2f}, required: {required_len:.2f}", flush=True)
        if path_len < required_len:
            _phase_b_print(f"[Phase B DEBUG] Path too short after clipping: {path_len:.2f} < {required_len:.2f}", flush=True)
            return None, "no internal path"
        window = propose_path_window(path, required_len, boundary_poly=safe_poly)
        if window is None or window.is_empty:
            _phase_b_print("[Phase B DEBUG] propose_path_window returned None or empty", flush=True)
            return None, "window not found"
        window_len = path_length(window)
        _phase_b_print(f"[Phase B DEBUG] Selected window length: {window_len:.2f}, required: {required_len:.2f}", flush=True)
        if window_len < required_len * 0.99:
            _phase_b_print(f"[Phase B DEBUG] Window too short: {window_len:.2f} < {required_len * 0.99:.2f}", flush=True)
            return None, "window not found"
        
        # Debug: Check if window vertices are inside before clipping
        outside_before = []
        for i, (x, y) in enumerate(window.coords):
            p = Point(x, y)
            if not (safe_poly.covers(p) or safe_poly.contains(p)):
                dist = safe_poly.boundary.distance(p)
                outside_before.append((i, x, y, dist))
        if outside_before:
            _phase_b_print(f"[Phase B DEBUG] {len(outside_before)} window vertices outside before clipping: {outside_before[:3]}", flush=True)

        # Clip window to safe polygon to ensure all segments (and interpolated points) are inside
        # Snapping vertices isn't enough - interpolation along segments can still go outside
        try:
            # Try buffering safe_poly slightly inward for precision tolerance
            try:
                safe_buffered = safe_poly.buffer(-0.1)
                if safe_buffered.is_empty:
                    safe_buffered = safe_poly
                logger.debug(f"Phase B: buffered safe_poly for clipping (buffer=-0.1)")
            except Exception as e:
                logger.debug(f"Phase B: buffer failed: {e}, using original safe_poly")
                safe_buffered = safe_poly
            
            clipped = window.intersection(safe_buffered)
            _phase_b_print(f"[Phase B DEBUG] Intersection result: type={type(clipped).__name__}, empty={clipped.is_empty}", flush=True)
            if clipped.is_empty:
                _phase_b_print("[Phase B DEBUG] Intersection is empty - window completely outside safe_poly", flush=True)
                return None, "window points outside safe polygon"
            
            # Handle intersection result (could be LineString, MultiLineString, or GeometryCollection)
            from shapely.geometry import MultiLineString, GeometryCollection
            if isinstance(clipped, LineString):
                window = clipped
            elif isinstance(clipped, MultiLineString) and clipped.geoms:
                # Take longest segment from MultiLineString
                best = None
                best_len = -1.0
                for g in clipped.geoms:
                    if isinstance(g, LineString) and not g.is_empty:
                        L = g.length
                        if L > best_len:
                            best_len = L
                            best = g
                if best is None or path_length(best) < required_len * 0.9:
                    return None, "window points outside safe polygon"
                window = best
            elif isinstance(clipped, GeometryCollection):
                # Extract LineStrings from collection
                lines = [g for g in clipped.geoms if isinstance(g, LineString) and not g.is_empty]
                if not lines:
                    return None, "window points outside safe polygon"
                window = max(lines, key=lambda x: x.length)
            else:
                return None, "window points outside safe polygon"
            
            # Re-check length after clipping (clipping might shorten the path)
            final_len = path_length(window)
            _phase_b_print(f"[Phase B DEBUG] Window after clipping: length={final_len:.2f}, required min={required_len * 0.9:.2f}", flush=True)
            if final_len < required_len * 0.9:
                _phase_b_print(f"[Phase B DEBUG] Window too short after clipping: {final_len:.2f} < {required_len * 0.9:.2f}", flush=True)
                return None, "window not found"
        except Exception as e:
            logger.warning(f"Phase B: clipping exception: {type(e).__name__}: {e}")
            # Fallback: if clipping fails, try snapping (less reliable)
            try:
                window_coords = list(window.coords)
                snapped_coords = _snap_coords_inside(safe_poly, window_coords, inset=0.5)
                if len(snapped_coords) < 2:
                    logger.warning("Phase B: snapping produced <2 points")
                    return None, "window not found"
                window = LineString(snapped_coords)
                snap_len = path_length(window)
                logger.debug(f"Phase B: fallback snapping, length={snap_len:.2f}")
                if snap_len < required_len * 0.95:
                    logger.warning(f"Phase B: snapped window too short ({snap_len:.2f} < {required_len * 0.95:.2f})")
                    return None, "window not found"
            except Exception as snap_err:
                logger.error(f"Phase B: snapping fallback also failed: {type(snap_err).__name__}: {snap_err}")
                return None, f"window clipping failed: {type(e).__name__}"

        # 1) HARD REQUIRE: window points must be inside safe polygon
        # Sample points along the clipped window
        step_pt = max(1.0, float(PATH_SAMPLE_STEP_PT))
        _phase_b_print(f"[Phase B DEBUG] Sampling points along window with step={step_pt:.2f}", flush=True)
        pts = sample_along_path(window, step_pt)
        if not pts:
            _phase_b_print("[Phase B DEBUG] sample_along_path returned no points", flush=True)
            return None, "window not found"
        _phase_b_print(f"[Phase B DEBUG] Sampled {len(pts)} points along window", flush=True)
        
        # Verify all sampled points are inside, and snap any outliers
        from shapely.ops import nearest_points
        outside_pts = []
        for i, p in enumerate(pts):
            if not (safe_poly.covers(p) or safe_poly.contains(p)):
                dist = safe_poly.boundary.distance(p)
                outside_pts.append((i, p.x, p.y, dist))
        
        if outside_pts:
            warn_msg = f"Phase B: {len(outside_pts)} sampled points outside safe_poly (out of {len(pts)})"
            logger.warning(warn_msg)
            logger.warning(f"Phase B: First few outside points: {outside_pts[:5]}")
            logger.warning(f"Phase B: Window coords (first 3): {list(window.coords[:3])}")
            logger.warning(f"Phase B: Safe poly bounds: {safe_poly.bounds}")
            # Print for visibility
            _phase_b_print(f"WARNING: {warn_msg}", flush=True)
            _phase_b_print(f"  Outside points (first 3): {outside_pts[:3]}", flush=True)
            
            # Snap any points that are outside (precision issue after clipping)
            # Use inner buffer to preserve clearance
            try:
                inner_safe = safe_poly.buffer(-0.5)
                if inner_safe.is_empty:
                    inner_safe = safe_poly
                    logger.debug("Phase B: inner buffer empty, using safe_poly")
                else:
                    logger.debug("Phase B: using inner buffer (-0.5) for snapping")
            except Exception as e:
                logger.debug(f"Phase B: inner buffer failed: {e}, using safe_poly")
                inner_safe = safe_poly
            
            snapped_pts = []
            snap_failed = 0
            for p in pts:
                if safe_poly.covers(p) or safe_poly.contains(p):
                    snapped_pts.append(p)
                else:
                    try:
                        # Snap to nearest point on inner buffer (preserves clearance)
                        q, _ = nearest_points(inner_safe, p)
                        # Verify the snapped point is actually inside safe_poly
                        if safe_poly.covers(q) or safe_poly.contains(q):
                            snapped_pts.append(q)
                        else:
                            # Fallback: snap to safe_poly boundary
                            q2, _ = nearest_points(safe_poly, p)
                            snapped_pts.append(q2)
                            logger.debug(f"Phase B: snapped point ({p.x:.2f}, {p.y:.2f}) -> ({q2.x:.2f}, {q2.y:.2f})")
                    except Exception as snap_err:
                        snap_failed += 1
                        snapped_pts.append(p)
                        logger.debug(f"Phase B: snapping failed for point ({p.x:.2f}, {p.y:.2f}): {snap_err}")
            
            if snap_failed > 0:
                _phase_b_print(f"[Phase B DEBUG] {snap_failed} points failed to snap", flush=True)
            
            _phase_b_print(f"[Phase B DEBUG] Snapped {len(snapped_pts)} points, checking if all inside...", flush=True)
            # Re-check with snapped points
            still_outside = []
            for p in snapped_pts:
                if not (safe_poly.covers(p) or safe_poly.contains(p)):
                    still_outside.append((p.x, p.y, safe_poly.boundary.distance(p)))
            
            if still_outside:
                error_msg = f"Phase B: {len(still_outside)} points still outside after snapping: {still_outside[:5]}"
                logger.error(error_msg)
                # Also print for visibility (CLI/Streamlit)
                _phase_b_print(f"[Phase B ERROR] {error_msg}", flush=True)
                _phase_b_print(f"[Phase B ERROR] Window coords count: {len(window.coords)}", flush=True)
                _phase_b_print(f"[Phase B ERROR] Window bounds: {window.bounds}", flush=True)
                _phase_b_print(f"[Phase B ERROR] Safe poly bounds: {safe_poly.bounds}", flush=True)
                return None, f"window points outside safe polygon ({len(still_outside)}/{len(snapped_pts)} points still outside)"
            else:
                _phase_b_print(f"[Phase B DEBUG] SUCCESS: All {len(snapped_pts)} points inside after snapping", flush=True)
        else:
            _phase_b_print(f"[Phase B DEBUG] SUCCESS: All {len(pts)} sampled points are inside safe_poly", flush=True)

        # 2) Clearance: measure against SAFE boundary (which already has padding applied)
        # So we only need CURVE_EXTRA_CLEARANCE_PT, not padding_pt + CURVE_EXTRA_CLEARANCE_PT
        _phase_b_print(f"[Phase B DEBUG] Checking clearance...", flush=True)
        min_clearance = float(min_clearance_along_path(window, safe_poly))
        required_clearance = float(max(0.0, CURVE_EXTRA_CLEARANCE_PT))
        _phase_b_print(f"[Phase B DEBUG] Clearance: min={min_clearance:.2f}, required={required_clearance:.2f}", flush=True)
        if min_clearance < required_clearance:
            _phase_b_print(f"[Phase B DEBUG] FAILED: Insufficient clearance", flush=True)
            return None, f"insufficient clearance: min={min_clearance:.2f}, required={required_clearance:.2f}"
        _phase_b_print(f"[Phase B DEBUG] Clearance check passed", flush=True)

        coords = list(window.coords)
        path_pt = _downsample_path(coords, CURVE_MAX_POINTS)

        # Anchor in the middle of the path (centroid is okay for labeling metadata)
        centroid = window.centroid
        anchor_pt = (float(centroid.x), float(centroid.y))

        curvature_deg = _curvature_total_deg(window)
        straightness = _straightness_ratio(window)
        # fit_margin relative to extra clearance requirement (not padding, since safe_poly already has padding)
        fit_margin = (min_clearance / required_clearance) if required_clearance > 0 else 1.0

        b = window.bounds
        bbox_pt = [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]

        _phase_b_print(f"[Phase B DEBUG] SUCCESS: Phase B curved placement completed!", flush=True)
        _phase_b_print(f"[Phase B DEBUG] Final window length: {path_length(window):.2f}, clearance: {min_clearance:.2f}", flush=True)
        
        return (
            PlacementResult(
                label_text=label_spec.text,
                font_size_pt=label_spec.font_size_pt,
                font_family=label_spec.font_family,
                geometry_source=geometry_source or "unknown",
                units="pt",
                mode="phase_b_curved",
                confidence=float(min(1.0, 0.55 + 0.25 * fit_margin + 0.05 * straightness)),
                anchor_pt=anchor_pt,
                angle_deg=0.0,
                bbox_pt=bbox_pt,
                path_pt=path_pt,
                min_clearance_pt=min_clearance,
                fit_margin_ratio=float(fit_margin),
                curvature_total_deg=float(curvature_deg),
                straightness_ratio=float(straightness),
                warnings=[],
            ),
            "",
        )
    except Exception as e:
        return None, f"exception: {type(e).__name__}"
    