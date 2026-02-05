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
    Write self-contained SVG: polygon, path for textPath, text with halo (stroke then fill).
    No external assets. Units in pt (geometry units). Never raises.
    """
    try:
        if not path_coords:
            Path(out_path).write_text("", encoding="utf-8")
            return
    except Exception:
        return

    path_d = _path_coords_to_svg_d(path_coords)
    poly_d = _poly_to_svg_d(polygon) if polygon and not polygon.is_empty else ""

    # Bounding box with margin
    all_x = [c[0] for c in path_coords]
    all_y = [c[1] for c in path_coords]
    try:
        if polygon and not polygon.is_empty and polygon.bounds:
            b = polygon.bounds
            all_x.extend([b[0], b[2]])
            all_y.extend([b[1], b[3]])
    except Exception:
        pass
    if not all_x:
        all_x = [0]
    if not all_y:
        all_y = [0]
    min_x = min(all_x) - 20
    max_x = max(all_x) + 20
    min_y = min(all_y) - 20
    max_y = max(all_y) + 20
    w = max(1, max_x - min_x)
    h = max(1, max_y - min_y)

    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    root = ET.Element("svg", {
        "xmlns": ns,
        "xmlns:xlink": "http://www.w3.org/1999/xlink",
        "width": f"{w:.2f}pt",
        "height": f"{h:.2f}pt",
        "viewBox": f"{min_x:.2f} {min_y:.2f} {w:.2f} {h:.2f}",
    })

    if poly_d:
        g_poly = ET.SubElement(root, "g", {"id": "polygon"})
        ET.SubElement(g_poly, "path", {
            "d": poly_d,
            "fill": "lightblue",
            "stroke": "navy",
            "stroke-width": "1",
        })

    defs = ET.SubElement(root, "defs")
    path_el = ET.SubElement(defs, "path", {
        "id": "riverpath",
        "d": path_d,
    })

    g_text = ET.SubElement(root, "g", {"id": "label"})
    # Halo: white stroke first
    text_stroke = ET.SubElement(g_text, "text", {
        "font-family": label_spec.font_family,
        "font-size": f"{label_spec.font_size_pt:.2f}pt",
        "fill": "none",
        "stroke": "white",
        "stroke-width": "2",
    })
    tpath_s = ET.SubElement(text_stroke, "textPath", {
        "{http://www.w3.org/1999/xlink}href": "#riverpath",
        "startOffset": "50%",
        "text-anchor": "middle",
    })
    tpath_s.text = label_spec.text

    text_fill = ET.SubElement(g_text, "text", {
        "font-family": label_spec.font_family,
        "font-size": f"{label_spec.font_size_pt:.2f}pt",
        "fill": "black",
        "stroke": "none",
    })
    tpath_f = ET.SubElement(text_fill, "textPath", {
        "{http://www.w3.org/1999/xlink}href": "#riverpath",
        "startOffset": "50%",
        "text-anchor": "middle",
    })
    tpath_f.text = label_spec.text

    try:
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        out_str = ET.tostring(root, encoding="unicode", method="xml")
        Path(out_path).write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + out_str, encoding="utf-8")
    except Exception:
        pass
