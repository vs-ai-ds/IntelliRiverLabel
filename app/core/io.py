# app/core/io.py
"""
Load and validate river geometry from WKT.
Supports Polygon, MultiPolygon, GeometryCollection (polygons extracted).
Invalid geometry is fixed with buffer(0) when possible.
See: docs/PROJECT_SPEC.md, docs/ARCHITECTURE.md.
"""

from __future__ import annotations

from pathlib import Path

from shapely import wkt
from shapely.geometry import (
    GeometryCollection,
    MultiPolygon,
    Polygon,
)
from shapely.geometry.base import BaseGeometry


def _resolve_path(path: str | Path, repo_root: Path | None) -> Path:
    """Resolve path; if relative, against repo_root (or cwd if repo_root is None)."""
    p = Path(path)
    if not p.is_absolute() and repo_root is not None:
        p = repo_root / p
    return p.resolve()


def load_wkt(path: str | Path, repo_root: Path | None = None) -> str:
    """
    Read WKT string from a file.
    Path can be overridden via CLI/UI; repo_root used for repo-relative paths.
    See: docs/PROJECT_SPEC.md (inputs).
    """
    resolved = _resolve_path(path, repo_root)
    if not resolved.exists():
        raise FileNotFoundError(f"Geometry file not found: {resolved}")
    return resolved.read_text(encoding="utf-8").strip()


def _first_wkt_only(wkt_string: str) -> str:
    """
    Return the first complete WKT geometry, stripping trailing text (e.g. "2." or extra lines).
    Handles GEOSException "Unexpected text after end of geometry".
    """
    s = wkt_string.strip()
    for prefix in ("MULTIPOLYGON", "POLYGON", "GEOMETRYCOLLECTION"):
        if s.upper().startswith(prefix):
            start = s.upper().index(prefix)
            i = start + len(prefix)
            depth = 0
            in_paren = False
            end = -1
            while i < len(s):
                c = s[i]
                if c == "(":
                    depth += 1
                    in_paren = True
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
                i += 1
            if end > 0:
                return s[start:end].strip()
            break
    return s


def parse_wkt(wkt_string: str) -> BaseGeometry:
    """
    Parse WKT string into a Shapely geometry.
    If the string contains extra text after the first geometry (e.g. "2."), only the first geometry is parsed.
    Does not fix validity; use validate_geometry for that.
    """
    single = _first_wkt_only(wkt_string)
    return wkt.loads(single)


def _extract_polygons(geom: BaseGeometry) -> list[Polygon]:
    """Extract one or more Polygon(s) from any supported geometry type."""
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        out: list[Polygon] = []
        for g in geom.geoms:
            out.extend(_extract_polygons(g))
        return out
    return []


def validate_geometry(geom: BaseGeometry) -> BaseGeometry:
    """
    Validate and fix geometry: ensure valid, return Polygon or MultiPolygon.
    Uses buffer(0) to fix invalid polygons when possible.
    See: docs/ARCHITECTURE.md (io.py).
    """
    if geom is None or geom.is_empty:
        raise ValueError("Geometry is empty or None")
    polygons = _extract_polygons(geom)
    if not polygons:
        raise ValueError("No polygon(s) found in geometry")
    fixed = [p.buffer(0) if not p.is_valid else p for p in polygons]
    fixed = [p for p in fixed if not p.is_empty and isinstance(p, Polygon)]
    if not fixed:
        raise ValueError("Geometry became empty after validation/fix")
    if len(fixed) == 1:
        return fixed[0]
    return MultiPolygon(fixed)


def load_and_validate_river(
    path: str | Path,
    repo_root: Path | None = None,
) -> BaseGeometry:
    """
    Load river WKT from file and return validated Polygon or MultiPolygon.
    Raises FileNotFoundError if path is missing, ValueError if geometry is invalid.
    See: docs/PROJECT_SPEC.md, docs/ARCHITECTURE.md.
    """
    wkt_string = load_wkt(path, repo_root)
    geom = parse_wkt(wkt_string)
    return validate_geometry(geom)
