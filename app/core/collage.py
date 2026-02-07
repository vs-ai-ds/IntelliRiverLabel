"""
Batch collage: grid of after.png thumbnails from batch cases.
Single file, no separate pipeline. See: IMPROVEMENTS_AND_SUGGESTIONS.md §6.2.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def make_batch_collage(batch_report: Path, max_images: int = 12, cols: int = 4) -> Path | None:
    """
    Create a grid image of first N after.png from batch cases. Saves to batch_report/collage.png.
    Returns path to collage or None if no images.
    """
    index_csv = batch_report / "index.csv"
    if not index_csv.exists():
        return None
    case_dirs: list[str] = []
    with open(index_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cd = (row.get("case_dir") or "").strip()
            if cd and not cd.startswith("_"):
                case_dirs.append(cd)
            if len(case_dirs) >= max_images:
                break
    if not case_dirs:
        return None
    images: list = []
    for cd in case_dirs:
        after = (batch_report / cd) / "after.png"
        if after.exists():
            try:
                images.append(mpimg.imread(str(after)))
            except Exception:
                pass
    if not images:
        return None
    n = len(images)
    ncols = min(cols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < len(images):
            ax.imshow(images[i])
    for j in range(len(images), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Batch preview (after.png)", fontsize=12)
    plt.tight_layout()
    out = batch_report / "collage.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out
