# app/ui/components.py
"""
Shared UI blocks for Demo, Debug, Export: header, images, metrics, warnings, downloads, SVG viewer.
Tabs assemble these with minimal extra logic. See: docs/UI_SPEC.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import streamlit as st

from app.ui.run_registry import RunRecord


def render_run_header(report_dir: Path, run_record: RunRecord | None = None, is_current: bool = False) -> None:
    """Title and optional badges (Curved / Fallback / Batch / Eval)."""
    name = report_dir.name
    if run_record:
        run_type = run_record.run_type
        mode = (run_record.metrics_summary or {}).get("mode") or ""
    else:
        run_type = "single"
        mode = ""
    if is_current:
        st.caption("**Current run**")
    st.caption(f"**{name}**")
    badges = []
    if run_type == "batch_case":
        badges.append("Batch case")
    elif run_type == "batch_root":
        badges.append("Batch")
    elif run_type == "eval":
        badges.append("Eval")
    if mode == "phase_b_curved":
        badges.append("Curved")
    elif mode and "phase" in str(mode).lower():
        badges.append("Fallback")
    if badges:
        st.caption(" · ".join(badges))


def render_before_after(report_dir: Path, full_width: bool = False) -> None:
    """Before/after images; full_width toggles layout."""
    before_path = report_dir / "before.png"
    after_path = report_dir / "after.png"
    if not before_path.exists() or not after_path.exists():
        return
    if full_width:
        st.image(str(before_path), caption="Before", width="stretch")
        st.image(str(after_path), caption="After", width="stretch")
    else:
        left, right = st.columns(2)
        with left:
            st.image(str(before_path), caption="Before")
        with right:
            st.image(str(after_path), caption="After")


def _centered_download(label: str, data: bytes, file_name: str, mime: str, key: str) -> None:
    """Render a download button centered in a 3-column layout."""
    _c1, _c2, _c3 = st.columns([1, 1, 1])
    with _c2:
        st.download_button(label, data=data, file_name=file_name, mime=mime, key=key)


def render_before_after_with_downloads(report_dir: Path, full_width: bool = False, key_prefix: str = "dl") -> None:
    """Before/after images with a centered download button below each (no debug)."""
    before_path = report_dir / "before.png"
    after_path = report_dir / "after.png"
    if not before_path.exists() or not after_path.exists():
        return
    if full_width:
        st.image(str(before_path), caption="Before", width="stretch")
        if before_path.exists():
            _centered_download("Download before.png", before_path.read_bytes(), "before.png", "image/png", f"{key_prefix}_before_png")
        st.image(str(after_path), caption="After", width="stretch")
        if after_path.exists():
            _centered_download("Download after.png", after_path.read_bytes(), "after.png", "image/png", f"{key_prefix}_after_png")
    else:
        left, right = st.columns(2)
        with left:
            st.image(str(before_path), caption="Before")
            if before_path.exists():
                _centered_download("Download before.png", before_path.read_bytes(), "before.png", "image/png", f"{key_prefix}_before_png")
        with right:
            st.image(str(after_path), caption="After")
            if after_path.exists():
                _centered_download("Download after.png", after_path.read_bytes(), "after.png", "image/png", f"{key_prefix}_after_png")


def render_metrics(metrics: dict) -> None:
    """DataFrame of metrics (from placement.json metrics + result.mode). All values as string for Arrow compatibility."""
    if not metrics:
        return
    rows = [{"Metric": str(k), "Value": "" if v is None else str(v)} for k, v in metrics.items()]
    import pandas as pd
    df = pd.DataFrame(rows)
    df = df.astype(str)
    st.dataframe(df, width="stretch", hide_index=True)


def render_warnings(warnings: list[str]) -> None:
    """One st.warning per warning."""
    for w in warnings:
        st.warning(w)


def render_placement_warnings(report_dir: Path) -> list[str]:
    """Read warnings from placement.json; return list. Caller may show only fallback reason in Demo."""
    p = report_dir / "placement.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("warnings", [])
    except Exception:
        return []


def render_download_buttons(report_dir: Path, key_prefix: str = "dl") -> None:
    """Download buttons for before.png, after.png, debug.png (when files exist)."""
    for name, fname in [("before.png", "before.png"), ("after.png", "after.png"), ("debug.png", "debug.png")]:
        path = report_dir / fname
        if path.exists():
            st.download_button(
                f"Download {name}",
                data=path.read_bytes(),
                file_name=name,
                mime="image/png",
                key=f"{key_prefix}_{name.replace('.', '_')}",
            )


def render_export_zip_and_files(report_dir: Path) -> None:
    """One section: individual file downloads, then Download all (zip) with caption below."""
    import io
    import zipfile
    from datetime import datetime

    def _file_info(p: Path) -> tuple[str, str]:
        if not p.exists():
            return "—", "—"
        try:
            stat = p.stat()
            size_str = f"{stat.st_size:,} B"
            mtime_str = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            return size_str, mtime_str
        except Exception:
            return "—", "—"

    for label, fname, mime in [
        ("placement.json", "placement.json", "application/json"),
        ("run_metadata.json", "run_metadata.json", "application/json"),
        ("before.png", "before.png", "image/png"),
        ("after.png", "after.png", "image/png"),
        ("debug.png", "debug.png", "image/png"),
        ("after.svg", "after.svg", "image/svg+xml"),
    ]:
        p = report_dir / fname
        if p.exists():
            size_str, mtime_str = _file_info(p)
            st.download_button(
                f"Download {label} ({size_str}, {mtime_str})",
                data=p.read_bytes(),
                file_name=label,
                mime=mime,
                key=f"dl_export_{label.replace('.', '_')}",
            )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ("placement.json", "run_metadata.json", "before.png", "after.png", "debug.png", "after.svg"):
            p = report_dir / name
            if p.exists():
                zf.write(p, name)
    zip_buffer.seek(0)
    st.download_button(
        "Download all (zip)",
        data=zip_buffer,
        file_name=f"{report_dir.name}_export.zip",
        mime="application/zip",
        key="dl_export_all_zip",
    )
    st.caption("Download all contains the files listed above in one zip (placement, metadata, images).")


def render_svg_viewer(svg_path: Path, height: int = 420) -> None:
    """Inline SVG viewer for curved label. Constrained size so it does not over-stretch."""
    if not svg_path.exists():
        return
    svg_text = svg_path.read_text(encoding="utf-8")
    if "<?xml" in svg_text:
        idx = svg_text.find("<svg")
        if idx >= 0:
            svg_text = svg_text[idx:]
    # Ensure SVG scales to fit without stretching (preserve aspect ratio)
    if "<svg" in svg_text and "style=" not in svg_text.split(">")[0]:
        svg_text = svg_text.replace("<svg", "<svg style=\"max-width:100%; height:auto;\"", 1)
    # Wrap: centered, scrollable, limited height
    wrapper = f'<div style="overflow:auto; max-height:{height}px; text-align:center;"><div style="display:inline-block; max-width:100%;">{svg_text}</div></div>'
    st.components.v1.html(wrapper, height=height + 20, scrolling=True)


def render_debug_image(debug_path: Path) -> None:
    """Debug overlay image or caption if missing. Image uses full width to avoid legend overlap."""
    if debug_path.exists():
        st.image(str(debug_path), width="stretch")
    else:
        st.caption("No debug image for this run. Run demo (with this run selected) to generate the overlay.")


def render_select_run_picker(
    records: list[RunRecord],
    caption: str,
    select_key: str,
    on_select_callback: Optional[Callable[[], None]] = None,
    default_path: Path | None = None,
    select_label: str = "Run",
) -> None:
    """Selectbox to choose a run (newest first). Selection is synced to active_ref in main flow (no st.rerun in callback)."""
    if not records:
        return
    options = [str(r.path) for r in records]
    default_resolved = default_path.resolve() if default_path else None
    index = next((i for i, r in enumerate(records) if r.path.resolve() == default_resolved), 0)
    if caption:
        st.caption(caption)
    kwargs = {"options": options, "format_func": lambda x: Path(x).name, "index": index, "key": select_key}
    if on_select_callback is not None:
        kwargs["on_change"] = on_select_callback
    st.selectbox(select_label, **kwargs)
