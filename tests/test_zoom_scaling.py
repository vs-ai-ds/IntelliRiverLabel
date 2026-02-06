# tests/test_zoom_scaling.py
"""
Zoom scaling: deterministic font/padding per bucket index.
"""

from __future__ import annotations

import pytest

from app.core.zoom import parse_zoom_buckets, scale_for_bucket


def test_parse_zoom_buckets_empty_default() -> None:
    out = parse_zoom_buckets("")
    assert len(out) >= 1
    assert all(isinstance(x, int) for x in out)


def test_parse_zoom_buckets_custom() -> None:
    out = parse_zoom_buckets("10,12,14")
    assert out == [10, 12, 14]
    out = parse_zoom_buckets("0,1,2")
    assert out == [0, 1, 2]


def test_scale_for_bucket_deterministic() -> None:
    base_font = 12.0
    base_padding = 3.0
    # Bucket index 0 -> scale 1.0; index 1 -> 1 + factor; index 2 -> 1 + 2*factor
    f0, p0 = scale_for_bucket(0, base_font, base_padding)
    f1, p1 = scale_for_bucket(1, base_font, base_padding)
    f2, p2 = scale_for_bucket(2, base_font, base_padding)
    assert f0 == base_font
    assert p0 == base_padding
    assert f1 > f0
    assert f2 > f1
    assert p1 >= p0
    assert p2 >= p1
    # Same inputs -> same outputs (deterministic)
    f0b, p0b = scale_for_bucket(0, base_font, base_padding)
    assert f0b == f0 and p0b == p0
