# app/core/render.py
"""
Matplotlib PNG rendering: before.png, after.png, debug.png.
See: docs/ARCHITECTURE.md, docs/ALGORITHM.md (debug overlays).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry.base import BaseGeometry

from app.core.config import RENDER_HEIGHT_PX, RENDER_WIDTH_PX
from app.core.geometry import polygon_bounds
from app.core.types import CandidatePoint, PlacementResult


def set_axes_to_polygon(ax: plt.Axes, polygon: BaseGeometry, pad_frac: float = 0.05) -> None:
    """Set xlim/ylim from polygon bounds with margin; equal aspect; hide axes."""
    if polygon is None or polygon.is_empty:
        return
    minx, miny, maxx, maxy = polygon_bounds(polygon)
    dx = max(1.0, (maxx - minx) * pad_frac)
    dy = max(1.0, (maxy - miny) * pad_frac)
    ax.set_xlim(minx - dx, maxx + dx)
    ax.set_ylim(miny - dy, maxy + dy)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def _new_fig(width_px: int, height_px: int) -> tuple[plt.Figure, plt.Axes]:
    # No constrained_layout: we set axes limits from polygon bounds; layout would collapse
    fig = plt.figure(
        figsize=(width_px / 100.0, height_px / 100.0),
        dpi=100,
        constrained_layout=False,
    )
    ax = fig.add_axes([0, 0, 1, 1])  # full-canvas axes
    ax.axis("off")
    return fig, ax


def _draw_polygon(ax: plt.Axes, geom: BaseGeometry, fill_alpha: float = 1.0) -> None:
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        xy = np.array(geom.exterior.coords)
        ax.fill(xy[:, 0], xy[:, 1], facecolor="lightblue", edgecolor="navy", linewidth=1, alpha=fill_alpha)
    else:
        for g in getattr(geom, "geoms", []):
            xy = np.array(g.exterior.coords)
            ax.fill(xy[:, 0], xy[:, 1], facecolor="lightblue", edgecolor="navy", linewidth=1, alpha=fill_alpha)


def render_before(
    river_geom: BaseGeometry,
    output_path: str | Path,
    width_px: int = RENDER_WIDTH_PX,
    height_px: int = RENDER_HEIGHT_PX,
    scale: int = 1,
) -> None:
    """Render river only. scale multiplies output resolution (1x, 2x, 4x)."""
    w, h = width_px * scale, height_px * scale
    fig, ax = _new_fig(w, h)
    _draw_polygon(ax, river_geom, fill_alpha=1.0)
    set_axes_to_polygon(ax, river_geom, pad_frac=0.05)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*constrained_layout.*", category=UserWarning)
        fig.savefig(output_path, dpi=100, facecolor="white")
    plt.close(fig)


def render_after(
    river_geom: BaseGeometry,
    placement: PlacementResult,
    output_path: str | Path,
    width_px: int = RENDER_WIDTH_PX,
    height_px: int = RENDER_HEIGHT_PX,
    halo_pt: float = 1.5,
    scale: int = 1,
) -> None:
    """Render river with label. scale multiplies output resolution (1x, 2x, 4x)."""
    w, h = width_px * scale, height_px * scale
    fig, ax = _new_fig(w, h)
    _draw_polygon(ax, river_geom, fill_alpha=1.0)

    cx, cy = placement.anchor_pt

    if placement.mode == "phase_b_curved" and placement.path_pt:
        # PNG still renders straight label at anchor (SVG is the curved output).
        # Keeping it simple and stable for hackathon judging.
        angle = 0.0
    else:
        angle = placement.angle_deg

    # halo
    for dx in (-halo_pt, 0.0, halo_pt):
        for dy in (-halo_pt, 0.0, halo_pt):
            if dx == 0 and dy == 0:
                continue
            ax.text(
                cx + dx, cy + dy, placement.label_text,
                fontsize=placement.font_size_pt,
                fontfamily=placement.font_family,
                ha="center", va="center",
                color="white",
                rotation=angle,
                zorder=5,
            )
    # fill
    ax.text(
        cx, cy, placement.label_text,
        fontsize=placement.font_size_pt,
        fontfamily=placement.font_family,
        ha="center", va="center",
        color="black",
        rotation=angle,
        zorder=6,
    )

    set_axes_to_polygon(ax, river_geom, pad_frac=0.05)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*constrained_layout.*", category=UserWarning)
        fig.savefig(output_path, dpi=100, facecolor="white")
    plt.close(fig)


def render_after_multi(
    river_geom: BaseGeometry,
    placements: list[PlacementResult],
    output_path: str | Path,
    width_px: int = RENDER_WIDTH_PX,
    height_px: int = RENDER_HEIGHT_PX,
    halo_pt: float = 1.5,
    scale: int = 1,
) -> None:
    """Render river with multiple labels. scale multiplies output resolution (1x, 2x, 4x)."""
    w, h = width_px * scale, height_px * scale
    fig, ax = _new_fig(w, h)
    _draw_polygon(ax, river_geom, fill_alpha=1.0)

    for placement in placements:
        cx, cy = placement.anchor_pt
        angle = placement.angle_deg

        # halo
        for dx in (-halo_pt, 0.0, halo_pt):
            for dy in (-halo_pt, 0.0, halo_pt):
                if dx == 0 and dy == 0:
                    continue
                ax.text(
                    cx + dx, cy + dy, placement.label_text,
                    fontsize=placement.font_size_pt,
                    fontfamily=placement.font_family,
                    ha="center", va="center",
                    color="white",
                    rotation=angle,
                    zorder=5,
                )
        # fill
        ax.text(
            cx, cy, placement.label_text,
            fontsize=placement.font_size_pt,
            fontfamily=placement.font_family,
            ha="center", va="center",
            color="black",
            rotation=angle,
            zorder=6,
        )

    set_axes_to_polygon(ax, river_geom, pad_frac=0.05)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*constrained_layout.*", category=UserWarning)
        fig.savefig(output_path, dpi=100, facecolor="white")
    plt.close(fig)


def render_debug(
    river_geom: BaseGeometry,
    safe_poly: BaseGeometry,
    candidates: list[CandidatePoint],
    placement: PlacementResult,
    output_path: str | Path,
    width_px: int = RENDER_WIDTH_PX,
    height_px: int = RENDER_HEIGHT_PX,
    scale: int = 1,
    occupied: BaseGeometry | None = None,
) -> None:
    """Render debug overlay. scale multiplies output resolution (1x, 2x, 4x). If occupied, draw semi-transparent hatch."""
    w, h = width_px * scale, height_px * scale
    fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100, constrained_layout=False)
    # Leave bottom margin so legend does not overlap the image
    ax = fig.add_axes([0.05, 0.08, 0.9, 0.88])  # left, bottom, width, height
    ax.axis("off")
    _draw_polygon(ax, river_geom, fill_alpha=0.5)

    if occupied is not None and not occupied.is_empty:
        if occupied.geom_type == "Polygon":
            xy = np.array(occupied.exterior.coords)
            ax.fill(xy[:, 0], xy[:, 1], facecolor="none", edgecolor="orange", linewidth=1, hatch="//", alpha=0.5, label="occupied")
        else:
            for g in getattr(occupied, "geoms", []):
                xy = np.array(g.exterior.coords)
                ax.fill(xy[:, 0], xy[:, 1], facecolor="none", edgecolor="orange", linewidth=1, hatch="//", alpha=0.5)

    if safe_poly is not None and not safe_poly.is_empty:
        if safe_poly.geom_type == "Polygon":
            xy = np.array(safe_poly.exterior.coords)
            ax.plot(xy[:, 0], xy[:, 1], linestyle="--", linewidth=1, label="safe")
        else:
            for g in getattr(safe_poly, "geoms", []):
                xy = np.array(g.exterior.coords)
                ax.plot(xy[:, 0], xy[:, 1], linestyle="--", linewidth=1)

    if candidates:
        xs = [c.x for c in candidates]
        ys = [c.y for c in candidates]
        ax.scatter(xs, ys, s=8, alpha=0.7, label="candidates")

    if placement.bbox_pt:
        bbox = placement.bbox_pt
        xy = np.array(bbox + [bbox[0]])
        ax.plot(xy[:, 0], xy[:, 1], linewidth=2, label="placement")

    if placement.path_pt:
        path_xy = np.array(placement.path_pt)
        ax.plot(path_xy[:, 0], path_xy[:, 1], linewidth=2, label="path")

    set_axes_to_polygon(ax, river_geom, pad_frac=0.05)
    # Legend in the reserved bottom margin so it never overlaps the image
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*constrained_layout.*", category=UserWarning)
        fig.savefig(output_path, dpi=100, facecolor="white", bbox_inches="tight", bbox_extra_artists=[leg])
    plt.close(fig)

