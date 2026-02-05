# app/core/render.py
"""
Matplotlib PNG rendering: before.png, after.png (with halo/stroke), debug.png.
No SVG implementation yet (stub). See: docs/ARCHITECTURE.md, docs/ALGORITHM.md (debug overlays).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry.base import BaseGeometry

from app.core.config import RENDER_HEIGHT_PX, RENDER_WIDTH_PX
from app.core.types import CandidatePoint, PlacementResult


def render_before(
    river_geom: BaseGeometry,
    output_path: str | Path,
    width_px: int = RENDER_WIDTH_PX,
    height_px: int = RENDER_HEIGHT_PX,
) -> None:
    """Draw river polygon only. See: docs/PLACEMENT_SCHEMA.md (before.png)."""
    fig, ax = plt.subplots(1, 1, figsize=(width_px / 100.0, height_px / 100.0), dpi=100)
    if river_geom and not river_geom.is_empty:
        if river_geom.geom_type == "Polygon":
            xy = np.array(river_geom.exterior.coords)
            ax.fill(xy[:, 0], xy[:, 1], facecolor="lightblue", edgecolor="navy", linewidth=1)
        else:
            for g in river_geom.geoms:
                xy = np.array(g.exterior.coords)
                ax.fill(xy[:, 0], xy[:, 1], facecolor="lightblue", edgecolor="navy", linewidth=1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def render_after(
    river_geom: BaseGeometry,
    placement: PlacementResult,
    output_path: str | Path,
    width_px: int = RENDER_WIDTH_PX,
    height_px: int = RENDER_HEIGHT_PX,
    halo_pt: float = 1.5,
) -> None:
    """
    Draw river + label with halo/stroke. See: docs/PLACEMENT_SCHEMA.md (after.png).
    """
    fig, ax = plt.subplots(1, 1, figsize=(width_px / 100.0, height_px / 100.0), dpi=100)
    if river_geom and not river_geom.is_empty:
        if river_geom.geom_type == "Polygon":
            xy = np.array(river_geom.exterior.coords)
            ax.fill(xy[:, 0], xy[:, 1], facecolor="lightblue", edgecolor="navy", linewidth=1)
        else:
            for g in river_geom.geoms:
                xy = np.array(g.exterior.coords)
                ax.fill(xy[:, 0], xy[:, 1], facecolor="lightblue", edgecolor="navy", linewidth=1)

    cx, cy = placement.anchor_pt
    angle = placement.angle_deg
    for dx in [-halo_pt, 0, halo_pt]:
        for dy in [-halo_pt, 0, halo_pt]:
            if dx == 0 and dy == 0:
                continue
            ax.text(cx + dx, cy + dy, placement.label_text, fontsize=placement.font_size_pt,
                    fontfamily=placement.font_family, ha="center", va="center", color="white",
                    rotation=angle, zorder=0)
    ax.text(cx, cy, placement.label_text, fontsize=placement.font_size_pt,
            fontfamily=placement.font_family, ha="center", va="center", color="black",
            rotation=angle, zorder=1)

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def render_debug(
    river_geom: BaseGeometry,
    safe_poly: BaseGeometry,
    candidates: list[CandidatePoint],
    placement: PlacementResult,
    output_path: str | Path,
    width_px: int = RENDER_WIDTH_PX,
    height_px: int = RENDER_HEIGHT_PX,
) -> None:
    """
    Debug overlay: river, safe polygon, candidate points (by score), chosen rectangle.
    See: docs/ALGORITHM.md (debug overlays).
    """
    fig, ax = plt.subplots(1, 1, figsize=(width_px / 100.0, height_px / 100.0), dpi=100)
    if river_geom and not river_geom.is_empty:
        if river_geom.geom_type == "Polygon":
            xy = np.array(river_geom.exterior.coords)
            ax.fill(xy[:, 0], xy[:, 1], facecolor="lightblue", edgecolor="navy", alpha=0.5)
        else:
            for g in river_geom.geoms:
                xy = np.array(g.exterior.coords)
                ax.fill(xy[:, 0], xy[:, 1], facecolor="lightblue", edgecolor="navy", alpha=0.5)
    if safe_poly and not safe_poly.is_empty:
        if safe_poly.geom_type == "Polygon":
            xy = np.array(safe_poly.exterior.coords)
            ax.plot(xy[:, 0], xy[:, 1], "g--", linewidth=1, label="safe")
        else:
            for g in safe_poly.geoms:
                xy = np.array(g.exterior.coords)
                ax.plot(xy[:, 0], xy[:, 1], "g--", linewidth=1)
    if candidates:
        xs = [c.x for c in candidates]
        ys = [c.y for c in candidates]
        ax.scatter(xs, ys, c="gray", s=8, alpha=0.7, label="candidates")
    bbox = placement.bbox_pt
    if bbox:
        xy = np.array(bbox + [bbox[0]])
        ax.plot(xy[:, 0], xy[:, 1], "r-", linewidth=2, label="placement")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def render_svg_path(
    river_geom: BaseGeometry,
    placement: PlacementResult,
    output_path: str | Path,
) -> None:
    """Stub: SVG text-on-path not implemented yet. See: docs/ARCHITECTURE.md (Phase B)."""
    from pathlib import Path
    p = Path(output_path)
    p.write_text("<!-- SVG curved mode not implemented -->", encoding="utf-8")
