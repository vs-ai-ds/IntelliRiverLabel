#!/usr/bin/env python3
"""
Generate 100 diverse river WKT files for testing IntelliRiverLabel.

Categories:
1-20:   Simple rivers (straight, slight curves, varying widths)
21-40:  Meandering rivers (S-curves, complex bends)
41-55:  Diverging rivers (delta, bifurcation - one source splitting)
56-70:  Converging rivers (tributaries joining into one)
71-85:  Complex multi-part rivers (disconnected segments)
86-95:  Edge cases (very thin, very wide, with islands)
96-100: Stress tests (extreme shapes)
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "assets" / "test_rivers"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Base coordinates (similar scale to Problem_1_river.wkt)
BASE_X = 11800.0
BASE_Y = 24500.0


def save_wkt(filename: str, geom: Polygon | MultiPolygon) -> None:
    """Save geometry to WKT file."""
    path = OUTPUT_DIR / filename
    wkt = geom.wkt
    # Format nicely with line breaks
    wkt = wkt.replace("), (", "),\n         (")
    wkt = wkt.replace(")), ((", ")),\n((")
    path.write_text(wkt, encoding="utf-8")
    print(f"Created: {path.name}")


def create_centerline(
    points: List[Tuple[float, float]],
    smooth: bool = True,
    smooth_iterations: int = 2,
) -> LineString:
    """Create a centerline from points, optionally smoothed."""
    if smooth and len(points) > 2:
        pts = np.array(points)
        for _ in range(smooth_iterations):
            smoothed = np.copy(pts)
            for i in range(1, len(pts) - 1):
                smoothed[i] = 0.25 * pts[i-1] + 0.5 * pts[i] + 0.25 * pts[i+1]
            pts = smoothed
        points = [(float(p[0]), float(p[1])) for p in pts]
    return LineString(points)


def buffer_with_varying_width(
    line: LineString,
    widths: List[float],
) -> Polygon:
    """Buffer a line with varying width along its length."""
    if len(widths) == 1:
        return line.buffer(widths[0], cap_style=2, join_style=2)
    
    # Interpolate widths along the line
    coords = list(line.coords)
    n = len(coords)
    
    # Create segments with interpolated widths
    segments = []
    for i in range(n - 1):
        t1 = i / (n - 1)
        t2 = (i + 1) / (n - 1)
        w1 = np.interp(t1, np.linspace(0, 1, len(widths)), widths)
        w2 = np.interp(t2, np.linspace(0, 1, len(widths)), widths)
        avg_width = (w1 + w2) / 2
        
        seg = LineString([coords[i], coords[i + 1]])
        segments.append(seg.buffer(avg_width, cap_style=2, join_style=2))
    
    result = unary_union(segments)
    if isinstance(result, MultiPolygon):
        # Return largest polygon
        return max(result.geoms, key=lambda g: g.area)
    return result


def generate_simple_river(
    seed: int,
    length: float = 400.0,
    width: float = 15.0,
    curve_amount: float = 0.1,
) -> Polygon:
    """Generate a simple river with slight curves."""
    rng = np.random.default_rng(seed)
    
    n_points = 20
    points = []
    for i in range(n_points):
        t = i / (n_points - 1)
        x = BASE_X + t * length
        # Add slight random curves
        y = BASE_Y + curve_amount * length * math.sin(t * math.pi * 2) * rng.uniform(0.5, 1.5)
        y += rng.uniform(-5, 5)
        points.append((x, y))
    
    line = create_centerline(points)
    return line.buffer(width, cap_style=2, join_style=2)


def generate_meandering_river(
    seed: int,
    length: float = 500.0,
    width: float = 12.0,
    meander_amplitude: float = 80.0,
    meander_frequency: float = 3.0,
) -> Polygon:
    """Generate a meandering S-curve river."""
    rng = np.random.default_rng(seed)
    
    n_points = 40
    points = []
    for i in range(n_points):
        t = i / (n_points - 1)
        x = BASE_X + t * length
        # Sinusoidal meander pattern
        y = BASE_Y + meander_amplitude * math.sin(t * meander_frequency * math.pi)
        # Add noise
        y += rng.uniform(-10, 10)
        points.append((x, y))
    
    line = create_centerline(points, smooth=True, smooth_iterations=3)
    
    # Vary width slightly
    widths = [width * rng.uniform(0.8, 1.2) for _ in range(5)]
    return buffer_with_varying_width(line, widths)


def generate_diverging_river(
    seed: int,
    main_length: float = 300.0,
    branch_length: float = 200.0,
    main_width: float = 18.0,
    branch_width: float = 10.0,
    n_branches: int = 2,
    spread_angle: float = 30.0,
) -> Polygon | MultiPolygon:
    """Generate a river that splits into branches (delta pattern)."""
    rng = np.random.default_rng(seed)
    
    # Main river stem
    main_points = []
    n_main = 15
    for i in range(n_main):
        t = i / (n_main - 1)
        x = BASE_X + t * main_length
        y = BASE_Y + rng.uniform(-5, 5)
        main_points.append((x, y))
    
    main_line = create_centerline(main_points)
    main_poly = main_line.buffer(main_width, cap_style=2)
    
    # Branch point
    branch_start = main_points[-1]
    
    # Create branches
    branch_polys = []
    for b in range(n_branches):
        angle_offset = spread_angle * (b - (n_branches - 1) / 2)
        angle_rad = math.radians(angle_offset)
        
        branch_points = [branch_start]
        n_branch = 12
        for i in range(1, n_branch):
            t = i / (n_branch - 1)
            dx = t * branch_length * math.cos(angle_rad)
            dy = t * branch_length * math.sin(angle_rad) + rng.uniform(-10, 10)
            branch_points.append((branch_start[0] + dx, branch_start[1] + dy))
        
        branch_line = create_centerline(branch_points)
        # Taper the branch
        widths = [branch_width * (1 - 0.3 * t) for t in np.linspace(0, 1, 5)]
        branch_poly = buffer_with_varying_width(branch_line, widths)
        branch_polys.append(branch_poly)
    
    # Union all parts
    all_parts = [main_poly] + branch_polys
    result = unary_union(all_parts)
    
    if not result.is_valid:
        result = result.buffer(0)
    
    return result


def generate_converging_river(
    seed: int,
    tributary_length: float = 200.0,
    main_length: float = 250.0,
    tributary_width: float = 10.0,
    main_width: float = 16.0,
    n_tributaries: int = 2,
    spread_angle: float = 40.0,
) -> Polygon | MultiPolygon:
    """Generate tributaries that merge into a main river."""
    rng = np.random.default_rng(seed)
    
    # Confluence point
    confluence = (BASE_X + tributary_length, BASE_Y)
    
    # Create tributaries coming in
    tributary_polys = []
    for t_idx in range(n_tributaries):
        angle_offset = spread_angle * (t_idx - (n_tributaries - 1) / 2)
        angle_rad = math.radians(180 + angle_offset)  # Coming from the left
        
        trib_points = []
        n_pts = 12
        for i in range(n_pts):
            t = i / (n_pts - 1)
            # Start from far, end at confluence
            dx = (1 - t) * tributary_length * math.cos(angle_rad)
            dy = (1 - t) * tributary_length * math.sin(angle_rad) + rng.uniform(-8, 8)
            trib_points.append((confluence[0] + dx, confluence[1] + dy))
        
        trib_line = create_centerline(trib_points)
        # Widen towards confluence
        widths = [tributary_width * (0.7 + 0.3 * t) for t in np.linspace(0, 1, 5)]
        trib_poly = buffer_with_varying_width(trib_line, widths)
        tributary_polys.append(trib_poly)
    
    # Main river after confluence
    main_points = [confluence]
    n_main = 15
    for i in range(1, n_main):
        t = i / (n_main - 1)
        x = confluence[0] + t * main_length
        y = confluence[1] + rng.uniform(-5, 5)
        main_points.append((x, y))
    
    main_line = create_centerline(main_points)
    main_poly = main_line.buffer(main_width, cap_style=2)
    
    # Union all parts
    all_parts = tributary_polys + [main_poly]
    result = unary_union(all_parts)
    
    if not result.is_valid:
        result = result.buffer(0)
    
    return result


def generate_multipart_river(
    seed: int,
    n_parts: int = 3,
    part_length: float = 150.0,
    width: float = 12.0,
    gap: float = 50.0,
) -> MultiPolygon:
    """Generate disconnected river segments."""
    rng = np.random.default_rng(seed)
    
    parts = []
    current_x = BASE_X
    
    for p in range(n_parts):
        n_pts = 15
        points = []
        for i in range(n_pts):
            t = i / (n_pts - 1)
            x = current_x + t * part_length
            y = BASE_Y + p * 60 + rng.uniform(-20, 20) + 30 * math.sin(t * math.pi)
            points.append((x, y))
        
        line = create_centerline(points)
        part_width = width * rng.uniform(0.8, 1.2)
        poly = line.buffer(part_width, cap_style=2)
        parts.append(poly)
        
        current_x += part_length + gap
    
    return MultiPolygon(parts)


def generate_river_with_island(
    seed: int,
    length: float = 400.0,
    width: float = 25.0,
    island_size: float = 0.3,
) -> Polygon:
    """Generate a wide river with an island (hole)."""
    rng = np.random.default_rng(seed)
    
    # Main river
    n_pts = 25
    points = []
    for i in range(n_pts):
        t = i / (n_pts - 1)
        x = BASE_X + t * length
        y = BASE_Y + 20 * math.sin(t * math.pi * 1.5) + rng.uniform(-5, 5)
        points.append((x, y))
    
    line = create_centerline(points)
    river = line.buffer(width, cap_style=2)
    
    # Create island in the middle
    island_center_t = 0.5 + rng.uniform(-0.1, 0.1)
    island_x = BASE_X + island_center_t * length
    island_y = BASE_Y + 20 * math.sin(island_center_t * math.pi * 1.5)
    
    # Elliptical island
    island_pts = []
    for angle in np.linspace(0, 2 * math.pi, 20):
        ix = island_x + width * island_size * math.cos(angle) * rng.uniform(0.8, 1.2)
        iy = island_y + width * island_size * 0.6 * math.sin(angle) * rng.uniform(0.8, 1.2)
        island_pts.append((ix, iy))
    island_pts.append(island_pts[0])
    
    island = Polygon(island_pts)
    
    # Subtract island from river
    result = river.difference(island)
    if not result.is_valid:
        result = result.buffer(0)
    
    if isinstance(result, MultiPolygon):
        return max(result.geoms, key=lambda g: g.area)
    return result


def generate_thin_river(
    seed: int,
    length: float = 350.0,
    width: float = 5.0,
) -> Polygon:
    """Generate a very thin river (stress test for label fitting)."""
    rng = np.random.default_rng(seed)
    
    n_pts = 30
    points = []
    for i in range(n_pts):
        t = i / (n_pts - 1)
        x = BASE_X + t * length
        y = BASE_Y + 15 * math.sin(t * math.pi * 2) + rng.uniform(-3, 3)
        points.append((x, y))
    
    line = create_centerline(points, smooth=True)
    return line.buffer(width, cap_style=2)


def generate_wide_river(
    seed: int,
    length: float = 300.0,
    width: float = 50.0,
) -> Polygon:
    """Generate a very wide river."""
    rng = np.random.default_rng(seed)
    
    n_pts = 20
    points = []
    for i in range(n_pts):
        t = i / (n_pts - 1)
        x = BASE_X + t * length
        y = BASE_Y + rng.uniform(-10, 10)
        points.append((x, y))
    
    line = create_centerline(points)
    # Vary width
    widths = [width * rng.uniform(0.7, 1.3) for _ in range(6)]
    return buffer_with_varying_width(line, widths)


def generate_horseshoe_river(
    seed: int,
    radius: float = 100.0,
    width: float = 15.0,
    opening_deg: float = 60.0,
) -> Polygon:
    """Generate a horseshoe/oxbow shaped river."""
    rng = np.random.default_rng(seed)
    
    start_angle = math.radians(opening_deg / 2)
    end_angle = math.radians(360 - opening_deg / 2)
    
    n_pts = 40
    points = []
    for i in range(n_pts):
        t = i / (n_pts - 1)
        angle = start_angle + t * (end_angle - start_angle)
        r = radius + rng.uniform(-5, 5)
        x = BASE_X + radius + r * math.cos(angle)
        y = BASE_Y + r * math.sin(angle)
        points.append((x, y))
    
    line = create_centerline(points)
    return line.buffer(width, cap_style=2)


def generate_zigzag_river(
    seed: int,
    length: float = 400.0,
    width: float = 12.0,
    n_zigs: int = 5,
    amplitude: float = 40.0,
) -> Polygon:
    """Generate a sharp zigzag river."""
    rng = np.random.default_rng(seed)
    
    points = []
    for i in range(n_zigs * 2 + 1):
        t = i / (n_zigs * 2)
        x = BASE_X + t * length
        y = BASE_Y + (amplitude if i % 2 == 1 else -amplitude) * (1 if i > 0 else 0)
        y += rng.uniform(-5, 5)
        points.append((x, y))
    
    line = create_centerline(points, smooth=True, smooth_iterations=1)
    return line.buffer(width, cap_style=2, join_style=2)


def generate_spiral_river(
    seed: int,
    n_turns: float = 1.5,
    start_radius: float = 30.0,
    end_radius: float = 120.0,
    width: float = 10.0,
) -> Polygon:
    """Generate a spiral-shaped river."""
    rng = np.random.default_rng(seed)
    
    n_pts = 60
    points = []
    for i in range(n_pts):
        t = i / (n_pts - 1)
        angle = t * n_turns * 2 * math.pi
        r = start_radius + t * (end_radius - start_radius)
        r += rng.uniform(-3, 3)
        x = BASE_X + 150 + r * math.cos(angle)
        y = BASE_Y + r * math.sin(angle)
        points.append((x, y))
    
    line = create_centerline(points)
    return line.buffer(width, cap_style=2)


def generate_complex_delta(
    seed: int,
) -> Polygon | MultiPolygon:
    """Generate a complex river delta with multiple branches."""
    rng = np.random.default_rng(seed)
    
    # Main stem
    main_points = [(BASE_X, BASE_Y)]
    for i in range(1, 10):
        t = i / 9
        x = BASE_X + t * 200
        y = BASE_Y + rng.uniform(-10, 10)
        main_points.append((x, y))
    
    branch_start = main_points[-1]
    
    # First split
    all_polys = []
    main_line = create_centerline(main_points)
    all_polys.append(main_line.buffer(18, cap_style=2))
    
    # Two primary branches
    for primary in range(2):
        p_angle = 20 * (1 if primary == 0 else -1)
        p_angle_rad = math.radians(p_angle)
        
        p_points = [branch_start]
        for i in range(1, 8):
            t = i / 7
            dx = t * 150 * math.cos(p_angle_rad)
            dy = t * 150 * math.sin(p_angle_rad) + rng.uniform(-8, 8)
            p_points.append((branch_start[0] + dx, branch_start[1] + dy))
        
        p_line = create_centerline(p_points)
        all_polys.append(p_line.buffer(12, cap_style=2))
        
        # Secondary branches from each primary
        secondary_start = p_points[-1]
        for secondary in range(2):
            s_angle = p_angle + 25 * (1 if secondary == 0 else -1)
            s_angle_rad = math.radians(s_angle)
            
            s_points = [secondary_start]
            for i in range(1, 6):
                t = i / 5
                dx = t * 100 * math.cos(s_angle_rad)
                dy = t * 100 * math.sin(s_angle_rad) + rng.uniform(-5, 5)
                s_points.append((secondary_start[0] + dx, secondary_start[1] + dy))
            
            s_line = create_centerline(s_points)
            all_polys.append(s_line.buffer(7, cap_style=2))
    
    result = unary_union(all_polys)
    if not result.is_valid:
        result = result.buffer(0)
    return result


def generate_braided_river(
    seed: int,
    length: float = 400.0,
    width: float = 8.0,
    n_channels: int = 3,
) -> Polygon | MultiPolygon:
    """Generate a braided river with multiple interweaving channels."""
    rng = np.random.default_rng(seed)
    
    all_polys = []
    
    for ch in range(n_channels):
        n_pts = 25
        points = []
        base_offset = (ch - (n_channels - 1) / 2) * 25
        
        for i in range(n_pts):
            t = i / (n_pts - 1)
            x = BASE_X + t * length
            # Channels weave around each other
            phase = ch * math.pi / n_channels
            y = BASE_Y + base_offset + 20 * math.sin(t * math.pi * 3 + phase)
            y += rng.uniform(-8, 8)
            points.append((x, y))
        
        line = create_centerline(points, smooth=True)
        poly = line.buffer(width * rng.uniform(0.8, 1.2), cap_style=2)
        all_polys.append(poly)
    
    result = unary_union(all_polys)
    if not result.is_valid:
        result = result.buffer(0)
    return result


def main():
    """Generate all 100 test river files."""
    print(f"Generating rivers in: {OUTPUT_DIR}")
    print("=" * 50)
    
    file_num = 1
    
    # 1-20: Simple rivers
    print("\n[1-20] Simple rivers...")
    for i in range(20):
        river = generate_simple_river(
            seed=1000 + i,
            length=300 + i * 15,
            width=10 + i * 0.5,
            curve_amount=0.05 + i * 0.02,
        )
        save_wkt(f"river_{file_num:03d}_simple.wkt", river)
        file_num += 1
    
    # 21-40: Meandering rivers
    print("\n[21-40] Meandering rivers...")
    for i in range(20):
        river = generate_meandering_river(
            seed=2000 + i,
            length=400 + i * 20,
            width=10 + i * 0.4,
            meander_amplitude=50 + i * 5,
            meander_frequency=2 + i * 0.2,
        )
        save_wkt(f"river_{file_num:03d}_meandering.wkt", river)
        file_num += 1
    
    # 41-55: Diverging rivers (delta/bifurcation)
    print("\n[41-55] Diverging rivers (delta pattern)...")
    for i in range(15):
        n_branches = 2 + (i % 4)  # 2-5 branches
        river = generate_diverging_river(
            seed=3000 + i,
            main_length=250 + i * 10,
            branch_length=150 + i * 10,
            main_width=15 + i * 0.5,
            branch_width=8 + i * 0.3,
            n_branches=n_branches,
            spread_angle=25 + i * 2,
        )
        save_wkt(f"river_{file_num:03d}_diverging_{n_branches}branch.wkt", river)
        file_num += 1
    
    # 56-70: Converging rivers (tributaries)
    print("\n[56-70] Converging rivers (tributaries)...")
    for i in range(15):
        n_tribs = 2 + (i % 3)  # 2-4 tributaries
        river = generate_converging_river(
            seed=4000 + i,
            tributary_length=180 + i * 10,
            main_length=200 + i * 10,
            tributary_width=8 + i * 0.3,
            main_width=14 + i * 0.4,
            n_tributaries=n_tribs,
            spread_angle=35 + i * 2,
        )
        save_wkt(f"river_{file_num:03d}_converging_{n_tribs}trib.wkt", river)
        file_num += 1
    
    # 71-85: Multi-part disconnected rivers
    print("\n[71-85] Multi-part rivers...")
    for i in range(15):
        n_parts = 2 + (i % 4)  # 2-5 parts
        river = generate_multipart_river(
            seed=5000 + i,
            n_parts=n_parts,
            part_length=120 + i * 8,
            width=10 + i * 0.4,
            gap=40 + i * 5,
        )
        save_wkt(f"river_{file_num:03d}_multipart_{n_parts}seg.wkt", river)
        file_num += 1
    
    # 86-90: Rivers with islands
    print("\n[86-90] Rivers with islands...")
    for i in range(5):
        river = generate_river_with_island(
            seed=6000 + i,
            length=350 + i * 20,
            width=25 + i * 3,
            island_size=0.25 + i * 0.05,
        )
        save_wkt(f"river_{file_num:03d}_with_island.wkt", river)
        file_num += 1
    
    # 91-93: Very thin rivers
    print("\n[91-93] Thin rivers...")
    for i in range(3):
        river = generate_thin_river(
            seed=7000 + i,
            length=300 + i * 50,
            width=4 + i,
        )
        save_wkt(f"river_{file_num:03d}_thin.wkt", river)
        file_num += 1
    
    # 94-95: Very wide rivers
    print("\n[94-95] Wide rivers...")
    for i in range(2):
        river = generate_wide_river(
            seed=8000 + i,
            length=280 + i * 40,
            width=45 + i * 10,
        )
        save_wkt(f"river_{file_num:03d}_wide.wkt", river)
        file_num += 1
    
    # 96-97: Horseshoe/oxbow rivers
    print("\n[96-97] Horseshoe rivers...")
    for i in range(2):
        river = generate_horseshoe_river(
            seed=9000 + i,
            radius=80 + i * 30,
            width=12 + i * 3,
            opening_deg=50 + i * 20,
        )
        save_wkt(f"river_{file_num:03d}_horseshoe.wkt", river)
        file_num += 1
    
    # 98: Complex delta
    print("\n[98] Complex delta...")
    river = generate_complex_delta(seed=10000)
    save_wkt(f"river_{file_num:03d}_complex_delta.wkt", river)
    file_num += 1
    
    # 99: Braided river
    print("\n[99] Braided river...")
    river = generate_braided_river(seed=11000, n_channels=4)
    save_wkt(f"river_{file_num:03d}_braided.wkt", river)
    file_num += 1
    
    # 100: Spiral river
    print("\n[100] Spiral river...")
    river = generate_spiral_river(seed=12000)
    save_wkt(f"river_{file_num:03d}_spiral.wkt", river)
    file_num += 1
    
    print("\n" + "=" * 50)
    print(f"Generated {file_num - 1} river WKT files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
