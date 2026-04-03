#!/usr/bin/env python3
"""Generate Phase 4 charts: artifact histogram, technique popularity, compliance rate."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from typing import Any

BG = "#161b22"
TEXT = "#e6edf3"
TEXT_DIM = "#8b949e"
GREEN = "#3fb950"
RED = "#f85149"
YELLOW = "#d29922"
ACCENT = "#bc8cff"
BLUE = "#58a6ff"
BORDER = "#30363d"

TECHNIQUE_COLORS = {
    "neural": GREEN,
    "ttt": ACCENT,
    "cache": YELLOW,
    "hybrid": BLUE,
}


def _parse_created(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").date()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Chart 1: Artifact Size Histogram
# ---------------------------------------------------------------------------

def generate_artifact_histogram(prs: list[dict[str, Any]]) -> str:
    """Generate an SVG histogram of artifact sizes in MB."""
    sizes_mb = [
        p["artifact_bytes"] / 1_000_000
        for p in prs
        if p.get("artifact_bytes") and p["artifact_bytes"] > 0
    ]
    if not sizes_mb:
        return _empty_svg("No artifact size data available", 800, 360)

    # Buckets: 0-2, 2-4, 4-6, 6-8, 8-10, 10-12, 12-14, 14-16, 16+
    bucket_edges = list(range(0, 18, 2)) + [999]
    bucket_labels = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12", "12-14", "14-16", "16+"]
    counts = [0] * len(bucket_labels)
    for s in sizes_mb:
        placed = False
        for i in range(len(bucket_edges) - 1):
            if bucket_edges[i] <= s < bucket_edges[i + 1]:
                counts[i] = counts[i] + 1
                placed = True
                break
        if not placed:
            counts[-1] += 1

    w, h = 800, 360
    ml, mr, mt, mb = 64, 20, 40, 56
    pw = w - ml - mr
    ph = h - mt - mb
    max_count = max(counts) or 1
    bar_w = pw / len(counts) - 4

    lines = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Artifact size histogram" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{w}" height="{h}" rx="12" fill="{BG}" stroke="{BORDER}"/>',
        f'<text x="{ml}" y="24" fill="{TEXT}" font-size="16" font-weight="700">Artifact Size Distribution</text>',
        f'<text x="{ml}" y="38" fill="{TEXT_DIM}" font-size="11">{len(sizes_mb)} PRs with known artifact size. 16 MB cap shown as red line.</text>',
    ]

    # Y gridlines
    y_step = _nice_step(max_count, 5)
    val = 0
    while val <= max_count:
        y = mt + ph - (val / max_count) * ph
        lines.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="{BORDER}" stroke-width="1"/>')
        lines.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" fill="{TEXT_DIM}" font-size="11" text-anchor="end">{val}</text>')
        val += y_step

    # Bars
    for i, count in enumerate(counts):
        x = ml + i * (pw / len(counts)) + 2
        bar_h = (count / max_count) * ph if count > 0 else 0
        y = mt + ph - bar_h
        color = RED if i == len(counts) - 1 else ACCENT
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="3" opacity="0.85"/>')
        if count > 0:
            lines.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" fill="{TEXT}" font-size="10" text-anchor="middle">{count}</text>')
        lines.append(f'<text x="{x + bar_w / 2:.1f}" y="{mt + ph + 16:.1f}" fill="{TEXT_DIM}" font-size="10" text-anchor="middle">{bucket_labels[i]}</text>')

    # 16MB cap line
    cap_x = ml + 7 * (pw / len(counts)) + pw / len(counts)
    lines.append(f'<line x1="{cap_x:.1f}" x2="{cap_x:.1f}" y1="{mt}" y2="{mt + ph}" stroke="{RED}" stroke-width="2" stroke-dasharray="6 4" opacity="0.7"/>')
    lines.append(f'<text x="{cap_x + 4:.1f}" y="{mt + 14}" fill="{RED}" font-size="10">16 MB cap</text>')

    lines.append(f'<text x="{ml + pw / 2:.1f}" y="{h - 6:.1f}" fill="{TEXT_DIM}" font-size="11" text-anchor="middle">Artifact size (MB)</text>')
    lines.append('</svg>')
    return "".join(lines)


# ---------------------------------------------------------------------------
# Chart 2: Technique Popularity Over Time (stacked area)
# ---------------------------------------------------------------------------

def generate_technique_popularity(prs: list[dict[str, Any]]) -> str:
    """Generate a stacked area chart of technique types over time."""
    by_date: dict[date, Counter] = defaultdict(Counter)
    for pr in prs:
        d = _parse_created(pr.get("created"))
        ttype = pr.get("technique_type", "neural")
        if d and ttype:
            by_date[d][ttype] += 1

    if not by_date:
        return _empty_svg("No technique data available", 800, 360)

    sorted_dates = sorted(by_date.keys())
    start_date = sorted_dates[0]
    end_date = max(sorted_dates[-1], date.today())

    # Build cumulative daily counts
    types = ["neural", "ttt", "cache", "hybrid"]
    cumulative: dict[str, list[int]] = {t: [] for t in types}
    running: dict[str, int] = {t: 0 for t in types}
    all_dates: list[date] = []
    d = start_date
    while d <= end_date:
        all_dates.append(d)
        for t in types:
            running[t] += by_date.get(d, Counter()).get(t, 0)
            cumulative[t].append(running[t])
        d += timedelta(days=1)

    w, h = 800, 360
    ml, mr, mt, mb_margin = 64, 20, 40, 56
    pw = w - ml - mr
    ph = h - mt - mb_margin
    max_total = max(sum(cumulative[t][i] for t in types) for i in range(len(all_dates))) or 1

    def x_pos(idx: int) -> float:
        return ml + (idx / max(len(all_dates) - 1, 1)) * pw

    def y_pos(val: int) -> float:
        return mt + ph - (val / max_total) * ph

    lines = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Technique popularity" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{w}" height="{h}" rx="12" fill="{BG}" stroke="{BORDER}"/>',
        f'<text x="{ml}" y="24" fill="{TEXT}" font-size="16" font-weight="700">Technique Popularity Over Time</text>',
        f'<text x="{ml}" y="38" fill="{TEXT_DIM}" font-size="11">Cumulative PR count by technique type. {len(prs)} total PRs.</text>',
    ]

    # Stacked areas (bottom to top: neural, ttt, cache, hybrid)
    for layer_idx in range(len(types) - 1, -1, -1):
        color = TECHNIQUE_COLORS[types[layer_idx]]
        # Top edge: sum of this layer and all below
        top_points = []
        for i in range(len(all_dates)):
            stack = sum(cumulative[types[j]][i] for j in range(layer_idx + 1))
            top_points.append(f"{x_pos(i):.1f},{y_pos(stack):.1f}")
        # Bottom edge: sum of layers below
        bottom_points = []
        for i in range(len(all_dates) - 1, -1, -1):
            stack = sum(cumulative[types[j]][i] for j in range(layer_idx))
            bottom_points.append(f"{x_pos(i):.1f},{y_pos(stack):.1f}")
        polygon = " ".join(top_points + bottom_points)
        lines.append(f'<polygon points="{polygon}" fill="{color}" opacity="0.6"/>')

    # X-axis date labels
    tick_interval = 3 if len(all_dates) <= 30 else 7
    for i, d in enumerate(all_dates):
        if i % tick_interval == 0:
            x = x_pos(i)
            lines.append(f'<text x="{x:.1f}" y="{mt + ph + 16:.1f}" fill="{TEXT_DIM}" font-size="10" text-anchor="middle">{d.strftime("%b %-d")}</text>')

    # Y gridlines
    y_step = _nice_step(max_total, 5)
    val = 0
    while val <= max_total:
        y = y_pos(val)
        lines.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="{BORDER}" stroke-width="1" opacity="0.5"/>')
        lines.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" fill="{TEXT_DIM}" font-size="11" text-anchor="end">{val}</text>')
        val += y_step

    # Legend
    legend_y = h - 12
    legend_x = ml
    for t in types:
        label = t.upper() if t != "neural" else "Neural"
        if t == "ttt":
            label = "TTT"
        color = TECHNIQUE_COLORS[t]
        lines.append(f'<rect x="{legend_x}" y="{legend_y - 8}" width="12" height="12" fill="{color}" rx="2" opacity="0.8"/>')
        lines.append(f'<text x="{legend_x + 16}" y="{legend_y + 2}" fill="{TEXT}" font-size="11">{label}</text>')
        legend_x += len(label) * 8 + 32

    lines.append('</svg>')
    return "".join(lines)


# ---------------------------------------------------------------------------
# Chart 3: Compliance Rate (% ALIVE vs AT_RISK vs DEAD over time)
# ---------------------------------------------------------------------------

def generate_compliance_rate(prs: list[dict[str, Any]]) -> str:
    """Generate a stacked area chart showing compliance rate over time."""
    statuses_of_interest = {"ALIVE", "AT_RISK", "DEAD"}
    by_date: dict[date, Counter] = defaultdict(Counter)
    for pr in prs:
        d = _parse_created(pr.get("created"))
        status = pr.get("status")
        if d and status in statuses_of_interest:
            by_date[d][status] += 1

    if not by_date:
        return _empty_svg("No compliance data available", 800, 360)

    sorted_dates = sorted(by_date.keys())
    start_date = sorted_dates[0]
    end_date = max(sorted_dates[-1], date.today())

    status_order = ["ALIVE", "AT_RISK", "DEAD"]
    status_colors = {"ALIVE": GREEN, "AT_RISK": YELLOW, "DEAD": RED}
    cumulative: dict[str, list[int]] = {s: [] for s in status_order}
    running: dict[str, int] = {s: 0 for s in status_order}
    all_dates: list[date] = []
    d = start_date
    while d <= end_date:
        all_dates.append(d)
        for s in status_order:
            running[s] += by_date.get(d, Counter()).get(s, 0)
            cumulative[s].append(running[s])
        d += timedelta(days=1)

    w, h = 800, 360
    ml, mr, mt, mb_margin = 64, 20, 40, 56
    pw = w - ml - mr
    ph = h - mt - mb_margin
    max_total = max(sum(cumulative[s][i] for s in status_order) for i in range(len(all_dates))) or 1

    def x_pos(idx: int) -> float:
        return ml + (idx / max(len(all_dates) - 1, 1)) * pw

    def y_pos(val: int) -> float:
        return mt + ph - (val / max_total) * ph

    lines = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Compliance rate" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{w}" height="{h}" rx="12" fill="{BG}" stroke="{BORDER}"/>',
        f'<text x="{ml}" y="24" fill="{TEXT}" font-size="16" font-weight="700">Compliance Rate Over Time</text>',
    ]

    # Compute final percentages for subtitle
    total_final = sum(cumulative[s][-1] for s in status_order)
    if total_final > 0:
        alive_pct = cumulative["ALIVE"][-1] / total_final * 100
        lines.append(f'<text x="{ml}" y="38" fill="{TEXT_DIM}" font-size="11">{total_final} classified PRs. Currently {alive_pct:.0f}% ALIVE.</text>')

    # Stacked areas (bottom to top: ALIVE, AT_RISK, DEAD)
    for layer_idx in range(len(status_order) - 1, -1, -1):
        color = status_colors[status_order[layer_idx]]
        top_points = []
        for i in range(len(all_dates)):
            stack = sum(cumulative[status_order[j]][i] for j in range(layer_idx + 1))
            top_points.append(f"{x_pos(i):.1f},{y_pos(stack):.1f}")
        bottom_points = []
        for i in range(len(all_dates) - 1, -1, -1):
            stack = sum(cumulative[status_order[j]][i] for j in range(layer_idx))
            bottom_points.append(f"{x_pos(i):.1f},{y_pos(stack):.1f}")
        polygon = " ".join(top_points + bottom_points)
        lines.append(f'<polygon points="{polygon}" fill="{color}" opacity="0.6"/>')

    # X-axis
    tick_interval = 3 if len(all_dates) <= 30 else 7
    for i, day in enumerate(all_dates):
        if i % tick_interval == 0:
            x = x_pos(i)
            lines.append(f'<text x="{x:.1f}" y="{mt + ph + 16:.1f}" fill="{TEXT_DIM}" font-size="10" text-anchor="middle">{day.strftime("%b %-d")}</text>')

    # Y gridlines
    y_step = _nice_step(max_total, 5)
    val = 0
    while val <= max_total:
        y = y_pos(val)
        lines.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="{BORDER}" stroke-width="1" opacity="0.5"/>')
        lines.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" fill="{TEXT_DIM}" font-size="11" text-anchor="end">{val}</text>')
        val += y_step

    # Legend
    legend_y = h - 12
    legend_x = ml
    for s in status_order:
        color = status_colors[s]
        label = s.replace("_", "-")
        lines.append(f'<rect x="{legend_x}" y="{legend_y - 8}" width="12" height="12" fill="{color}" rx="2" opacity="0.8"/>')
        lines.append(f'<text x="{legend_x + 16}" y="{legend_y + 2}" fill="{TEXT}" font-size="11">{label}</text>')
        legend_x += len(label) * 8 + 32

    lines.append('</svg>')
    return "".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nice_step(max_val: int, target_ticks: int) -> int:
    """Compute a human-friendly step size for axis ticks."""
    if max_val <= 0:
        return 1
    raw = max_val / target_ticks
    magnitude = 10 ** math.floor(math.log10(raw))
    for candidate in (1, 2, 5, 10):
        step = candidate * magnitude
        if step >= raw:
            return max(int(step), 1)
    return max(int(magnitude * 10), 1)


def _empty_svg(message: str, width: int, height: int) -> str:
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="{message}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="{width}" height="{height}" fill="{BG}" rx="12"/>'
        f'<text x="{width / 2:.0f}" y="{height / 2:.0f}" fill="{TEXT}" font-size="18" text-anchor="middle">{message}</text>'
        '</svg>'
    )
