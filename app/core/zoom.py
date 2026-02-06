# app/core/zoom.py
"""
Zoom bucket parsing and font/padding scaling for consistent placement across zoom levels.
"""

from __future__ import annotations

from app.core.config import (
    PADDING_PT,
    ZOOM_BUCKETS_DEFAULT,
    ZOOM_FONT_SCALE_FACTOR,
    ZOOM_PADDING_SCALE_FACTOR,
)


def parse_zoom_buckets(s: str) -> list[int]:
    """Parse comma-separated zoom buckets, e.g. '10,12,14' or '0,1,2'. Returns list of ints."""
    if not (s or "").strip():
        return list(ZOOM_BUCKETS_DEFAULT)
    out: list[int] = []
    for part in s.strip().split(","):
        part = part.strip()
        if part:
            try:
                out.append(int(part))
            except ValueError:
                continue
    return out if out else list(ZOOM_BUCKETS_DEFAULT)


def scale_for_bucket(
    bucket: int,
    base_font_pt: float,
    base_padding_pt: float | None = None,
) -> tuple[float, float]:
    """
    Return (font_pt, padding_pt) scaled for the given bucket index.
    font_pt = base_font * (1 + ZOOM_FONT_SCALE_FACTOR * bucket)
    padding_pt = base_padding * (1 + ZOOM_PADDING_SCALE_FACTOR * bucket)
    """
    pad = base_padding_pt if base_padding_pt is not None else PADDING_PT
    font = base_font_pt * (1.0 + ZOOM_FONT_SCALE_FACTOR * bucket)
    padding = pad * (1.0 + ZOOM_PADDING_SCALE_FACTOR * bucket)
    return (max(1.0, font), max(0.5, padding))
