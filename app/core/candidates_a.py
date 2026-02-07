# app/core/candidates_a.py
"""
Phase A candidates: sample points in safe polygon, clearance, top K, angle candidates
from PCA + config deltas, upright orientation. Includes AI-powered K-means clustering
for smarter candidate distribution. See: docs/ALGORITHM.md A2–A3.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from app.core.config import (
    ANGLE_OFFSETS_DEG,
    K_TOP_CLEARANCE,
    N_SAMPLE_POINTS,
    SEED,
)
from app.core.geometry import pca_dominant_angle_deg, sample_points_in_polygon
from app.core.types import CandidatePoint


def _clearance_pt(geom: BaseGeometry, x: float, y: float) -> float:
    """Distance from (x, y) to boundary of geom."""
    if geom is None or geom.is_empty:
        return 0.0
    p = Point(x, y)
    return float(geom.boundary.distance(p))


def _smart_sample_with_kmeans(
    safe_polygon: BaseGeometry,
    n_sample: int,
    n_clusters: int = 8,
    seed: int | None = None,
) -> list[tuple[float, float]]:
    """
    AI-powered sampling: use K-means to find distinct regions, then sample
    the best points from each cluster. This ensures coverage of all "good" areas.
    """
    # First, dense random sampling
    raw_pts = sample_points_in_polygon(safe_polygon, n_sample * 2, seed=seed)
    if len(raw_pts) < n_clusters * 2:
        return raw_pts[:n_sample]
    
    try:
        from sklearn.cluster import KMeans
        
        X = np.array(raw_pts)
        
        # Add clearance as a feature for clustering (weight spatial + quality)
        clearances = np.array([_clearance_pt(safe_polygon, x, y) for x, y in raw_pts])
        clearances_norm = clearances / (clearances.max() + 1e-6)
        
        # Cluster on (x, y, clearance) to group spatially AND by quality
        X_extended = np.column_stack([X, clearances_norm * 50])  # Scale clearance
        
        actual_clusters = min(n_clusters, len(raw_pts) // 2)
        kmeans = KMeans(n_clusters=actual_clusters, random_state=seed or 42, n_init=10)
        labels = kmeans.fit_predict(X_extended)
        
        # From each cluster, pick points with highest clearance
        result = []
        samples_per_cluster = max(1, n_sample // actual_clusters)
        
        for cluster_id in range(actual_clusters):
            mask = labels == cluster_id
            cluster_indices = np.where(mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Sort by clearance within cluster
            cluster_clearances = clearances[cluster_indices]
            sorted_idx = np.argsort(cluster_clearances)[::-1]  # Descending
            
            for i in sorted_idx[:samples_per_cluster]:
                idx = cluster_indices[i]
                result.append(raw_pts[idx])
        
        return result[:n_sample]
        
    except ImportError:
        # Fallback if sklearn not available
        return raw_pts[:n_sample]


def generate_candidate_points(
    safe_polygon: BaseGeometry,
    n_sample: int = N_SAMPLE_POINTS,
    k_top: int = K_TOP_CLEARANCE,
    seed: int | None = SEED,
    use_smart_sampling: bool = True,
) -> list[CandidatePoint]:
    """
    Sample n points inside safe polygon, compute clearance, keep top K.
    When use_smart_sampling=True, uses AI-powered K-means clustering for
    better distribution across the polygon. See: docs/ALGORITHM.md A2.
    """
    if safe_polygon is None or safe_polygon.is_empty:
        return []
    
    if use_smart_sampling:
        pts = _smart_sample_with_kmeans(safe_polygon, n_sample, seed=seed)
    else:
        pts = sample_points_in_polygon(safe_polygon, n_sample, seed=seed)
    
    with_clearance = [
        (x, y, _clearance_pt(safe_polygon, x, y)) for x, y in pts
    ]
    with_clearance.sort(key=lambda t: t[2], reverse=True)
    top = with_clearance[:k_top]
    return [
        CandidatePoint(x=x, y=y, clearance=cl, base_score=cl, features={"clearance": cl})
        for x, y, cl in top
    ]


def angle_candidates_deg(
    geom: BaseGeometry,
    offsets_deg: tuple[float, ...] = ANGLE_OFFSETS_DEG,
) -> list[float]:
    """
    Base angle from PCA of boundary, then base ± each offset.
    Enforce upright: return angles in [0, 180) so text is not upside-down.
    See: docs/ALGORITHM.md A3.
    """
    base = pca_dominant_angle_deg(geom)
    angles: list[float] = []
    for off in offsets_deg:
        a = base + off
        a = a % 180.0
        if a < 0:
            a += 180.0
        if a not in angles:
            angles.append(a)
    for off in offsets_deg:
        a = base - off
        a = a % 180.0
        if a < 0:
            a += 180.0
        if a not in angles:
            angles.append(a)
    return sorted(set(angles))
