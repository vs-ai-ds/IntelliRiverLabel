# app/core/text_metrics.py
"""
Measure text width/height in pt using Pillow. 1 pt = 1 geometry unit.
See: docs/ALGORITHM.md A4, docs/PROJECT_SPEC.md.
"""

from __future__ import annotations

import warnings

_font_warning_emitted: set[str] = set()


def _load_font(font_family: str, font_size_pt: float):
    """Load PIL ImageFont; fallback with warning if font not found."""
    global _font_warning_emitted
    from PIL import ImageFont

    size = max(1, int(round(font_size_pt)))
    candidates = [
        font_family + ".ttf",
        font_family.replace(" ", "") + ".ttf",
        "DejaVuSans.ttf",
        "arial.ttf",
        "Arial.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except (OSError, IOError):
            continue
    if font_family not in _font_warning_emitted:
        _font_warning_emitted.add(font_family)
        warnings.warn(f"Font not found: {font_family!r}; using default.", UserWarning)
    return ImageFont.load_default()


def measure_text_pt(text: str, font_family: str, font_size_pt: float) -> tuple[float, float]:
    """
    Return (width_pt, height_pt). 1 pt = 1 geometry unit.
    Uses Pillow; fallback font with warning if requested font not found.
    """
    from PIL import Image, ImageDraw

    font = _load_font(font_family, font_size_pt)
    img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = float(bbox[2] - bbox[0])
    h = float(bbox[3] - bbox[1])
    # At 72 DPI, 1 pt = 1 px. Font size was set in pt; bbox is in px ≈ pt.
    try:
        size_used = getattr(font, "size", font_size_pt)
    except Exception:
        size_used = font_size_pt
    scale = font_size_pt / max(1.0, float(size_used))
    return (w * scale, h * scale)
