# app/core/render_svg.py
"""
Export curved label as self-contained SVG: polygon, path, text-on-path with halo.
See: docs/ARCHITECTURE.md, docs/ALGORITHM.md B3.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from app.core.types import LabelSpec

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"


def _poly_to_svg_d(geom: BaseGeometry) -> str:
    """Single polygon exterior as SVG path d (M L ... Z)."""
    if geom is None or geom.is_empty:
        return ""
    if isinstance(geom, Polygon):
        coords = list(geom.exterior.coords)
        if len(coords) < 2:
            return ""
        parts = [f"M {coords[0][0]:.4f} {coords[0][1]:.4f}"]
        for x, y in coords[1:]:
            parts.append(f"L {x:.4f} {y:.4f}")
        parts.append("Z")
        return " ".join(parts)
    if isinstance(geom, MultiPolygon):
        return " ".join(_poly_to_svg_d(p) for p in geom.geoms if not p.is_empty)
    return ""


def _path_coords_to_svg_d(path_coords: list[tuple[float, float]]) -> str:
    """Polyline to SVG path d (M L L ...)."""
    if not path_coords:
        return ""
    parts = [f"M {path_coords[0][0]:.4f} {path_coords[0][1]:.4f}"]
    for x, y in path_coords[1:]:
        parts.append(f"L {x:.4f} {y:.4f}")
    return " ".join(parts)


def export_curved_svg(
    polygon: BaseGeometry,
    label_spec: LabelSpec,
    path_coords: list[tuple[float, float]],
    out_path: str | Path,
) -> None:
    """
    Write self-contained SVG: polygon, path for textPath, text with halo.
    Units in pt (geometry units). Never raises.
    """
    try:
        if not path_coords:
            Path(out_path).write_text("", encoding="utf-8")
            return
    except Exception:
        return

    path_d = _path_coords_to_svg_d(path_coords)
    poly_d = _poly_to_svg_d(polygon) if polygon and not polygon.is_empty else ""

    # viewBox bounds (geometry coords)
    try:
        if polygon and not polygon.is_empty and getattr(polygon, "bounds", None):
            min_x, min_y, max_x, max_y = map(float, polygon.bounds)
        else:
            xs = [c[0] for c in path_coords]
            ys = [c[1] for c in path_coords]
            min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)
    except Exception:
        xs = [c[0] for c in path_coords]
        ys = [c[1] for c in path_coords]
        min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)

    margin = 20.0
    min_x -= margin
    min_y -= margin
    max_x += margin
    max_y += margin
    vw = max(1.0, max_x - min_x)
    vh = max(1.0, max_y - min_y)

    # IMPORTANT:
    # - Use plain tag names (not {ns}svg) and set xmlns attributes exactly once
    # - Use prefixed attributes like "xlink:href" (not Clark-notation {xlink}href)
    root = ET.Element(
        "svg",
        {
            "xmlns": SVG_NS,
            "xmlns:xlink": XLINK_NS,
            "width": "800",
            "height": "600",
            "viewBox": f"{min_x:.2f} {min_y:.2f} {vw:.2f} {vh:.2f}",
            "preserveAspectRatio": "xMidYMid meet",
        },
    )

    # Correct Y-flip for non-zero viewBox origin:
    # y' = -y + (2*min_y + vh)
    flip_translate_y = (2.0 * min_y) + vh
    flip = ET.SubElement(
        root,
        "g",
        {"transform": f"translate(0 {flip_translate_y:.2f}) scale(1 -1)"},
    )

    if poly_d:
        ET.SubElement(
            flip,
            "path",
            {
                "d": poly_d,
                "fill": "lightblue",
                "stroke": "navy",
                "stroke-width": "1",
            },
        )

    defs = ET.SubElement(flip, "defs")
    ET.SubElement(defs, "path", {"id": "riverpath", "d": path_d})

    g_text = ET.SubElement(flip, "g", {"id": "label"})

    text = ET.SubElement(
        g_text,
        "text",
        {
            "font-family": label_spec.font_family,
            "font-size": f"{label_spec.font_size_pt:.2f}",
            "fill": "black",
            "stroke": "white",
            "stroke-width": "2.25",
            "paint-order": "stroke",
            "stroke-linejoin": "round",
            "stroke-linecap": "round",
        },
    )

    # Provide both SVG2 (href) and SVG1.1 (xlink:href) without triggering xmlns duplication
    tpath = ET.SubElement(
        text,
        "textPath",
        {
            "href": "#riverpath",
            "xlink:href": "#riverpath",
            "startOffset": "50%",
            "text-anchor": "middle",
        },
    )
    tpath.text = label_spec.text

    try:
        out_str = ET.tostring(root, encoding="unicode", method="xml")
        Path(out_path).write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + out_str, encoding="utf-8")
    except Exception:
        pass
    