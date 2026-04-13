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


# ---------------------------------------------------------------------------
# Chart 4: Community Activity Timeline (dual-axis bar + line)
# ---------------------------------------------------------------------------

def generate_community_activity(csv_path: str = "") -> str:
    """Generate an SVG chart showing community activity with Agora review overlay.

    Reads data/community_activity.csv with columns:
    date, mato_comments, other_comments, prs_created, prs_updated
    """
    import csv
    from pathlib import Path

    if not csv_path:
        csv_path = str(Path(__file__).parent.parent / "data" / "community_activity.csv")

    data_path = Path(csv_path)
    if not data_path.exists():
        return _empty_svg("No community activity data available", 800, 420)

    rows = []
    with data_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["date"].strip():
                rows.append(row)

    if not rows:
        return _empty_svg("No community activity data available", 800, 420)

    dates = [r["date"] for r in rows]
    mato = [int(r["mato_comments"]) for r in rows]
    other = [int(r["other_comments"]) for r in rows]
    prs_created = [int(r["prs_created"]) for r in rows]
    prs_updated = [int(r["prs_updated"]) for r in rows]

    w, h = 800, 420
    ml, mr, mt, mb = 64, 64, 40, 56
    pw = w - ml - mr
    ph = h - mt - mb
    n = len(dates)
    bar_w = max(pw / n - 2, 4)

    # Left axis: comments (ceil to nearest 200, min 800), Right axis: PRs
    raw_max_comments = max(max(m + o for m, o in zip(mato, other)), 1)
    max_comments = max(((raw_max_comments + 199) // 200) * 200, 800)
    # Right axis ceiling: match left axis proportions so dual-axis heights are comparable
    # e.g. if left goes to 800, right should go to ~800 too
    raw_max_prs = max(max(prs_created), max(prs_updated))
    max_prs = max(((raw_max_prs + 199) // 200) * 200 + 200, 400)

    def x_pos(i: int) -> float:
        return ml + (i + 0.5) * (pw / n)

    def y_left(val: float) -> float:
        return mt + ph - (val / max_comments) * ph

    def y_right(val: float) -> float:
        return mt + ph - (val / max_prs) * ph

    # Find the sweep days for annotation
    sweep_start = None
    for i, m in enumerate(mato):
        if m > 100:
            sweep_start = i
            break

    lines = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Community activity" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{w}" height="{h}" rx="12" fill="{BG}" stroke="{BORDER}"/>',
        f'<text x="{ml}" y="24" fill="{TEXT}" font-size="16" font-weight="700">Community Activity vs. Agora Reviews</text>',
        f'<text x="{ml}" y="38" fill="{TEXT_DIM}" font-size="11">Stacked bars = comments/day (left axis). Lines = new PRs/day + PR updates/day (right axis). Purple = Agora reviews.</text>',
    ]

    # Y gridlines (left axis)
    y_step = _nice_step(max_comments, 5)
    val = 0
    while val <= max_comments:
        y = y_left(val)
        lines.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="{BORDER}" stroke-width="1" opacity="0.4"/>')
        lines.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" fill="{TEXT_DIM}" font-size="10" text-anchor="end">{val}</text>')
        val += y_step

    # Right axis labels (PRs)
    y_step_r = _nice_step(max_prs, 5)
    val = 0
    while val <= max_prs:
        y = y_right(val)
        lines.append(f'<text x="{ml + pw + 8}" y="{y + 4:.1f}" fill="{YELLOW}" font-size="10" text-anchor="start">{val}</text>')
        val += y_step_r

    # Sweep highlight band
    if sweep_start is not None:
        sweep_x = ml + sweep_start * (pw / n)
        sweep_w = min(2, n - sweep_start) * (pw / n)
        lines.append(f'<rect x="{sweep_x:.1f}" y="{mt}" width="{sweep_w:.1f}" height="{ph}" fill="{ACCENT}" opacity="0.08"/>')

    # Stacked bars: other (bottom, blue) + mato (top, purple)
    for i in range(n):
        x = x_pos(i) - bar_w / 2

        # Other comments bar (bottom)
        if other[i] > 0:
            bh = (other[i] / max_comments) * ph
            by = y_left(other[i])
            lines.append(f'<rect x="{x:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{BLUE}" rx="2" opacity="0.7"/>')

        # Mato comments bar (stacked on top)
        if mato[i] > 0:
            base = other[i]
            bh = (mato[i] / max_comments) * ph
            by = y_left(base + mato[i])
            lines.append(f'<rect x="{x:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{ACCENT}" rx="2" opacity="0.85"/>')

    # PRs created line (yellow, drawn first = behind)
    points = []
    for i in range(n):
        points.append(f"{x_pos(i):.1f},{y_right(prs_created[i]):.1f}")
    lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{YELLOW}" stroke-width="2" opacity="0.8"/>')
    for i in range(n):
        lines.append(f'<circle cx="{x_pos(i):.1f}" cy="{y_right(prs_created[i]):.1f}" r="2" fill="{YELLOW}" opacity="0.8"/>')

    # PR updates line (green, drawn last = on top)
    points_upd = []
    for i in range(n):
        points_upd.append(f"{x_pos(i):.1f},{y_right(prs_updated[i]):.1f}")
    lines.append(f'<polyline points="{" ".join(points_upd)}" fill="none" stroke="{GREEN}" stroke-width="2.5" opacity="0.9"/>')
    for i in range(n):
        lines.append(f'<circle cx="{x_pos(i):.1f}" cy="{y_right(prs_updated[i]):.1f}" r="3" fill="{GREEN}" opacity="0.9"/>')

    # Interactive tooltip hover columns for each day
    def fmt_date(d: str) -> str:
        mn = {"01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May", "06": "Jun"}
        parts = d.split("-")
        return f"{mn.get(parts[1], parts[1])} {int(parts[2])}"

    # Vertical hover highlight + floating tooltip box (SVG-native)
    col_w = pw / n
    for i in range(n):
        cx = ml + i * col_w
        tip_lines = [
            fmt_date(dates[i]),
            f"Community: {other[i]}",
            f"Agora reviews: {mato[i]}",
            f"New PRs: {prs_created[i]}",
            f"PR updates: {prs_updated[i]}",
        ]
        # Tooltip background + text group, hidden by default
        tip_x = min(cx + col_w + 4, ml + pw - 140)
        tip_y = mt + 10
        gid = f"tip-{i}"
        lines.append(f'<g id="{gid}" visibility="hidden" pointer-events="none">')
        lines.append(f'<rect x="{tip_x:.0f}" y="{tip_y}" width="138" height="82" rx="6" fill="#1c2128" stroke="{BORDER}" opacity="0.95"/>')
        for j, tl in enumerate(tip_lines):
            ty = tip_y + 15 + j * 14
            color = TEXT if j == 0 else [BLUE, ACCENT, YELLOW, GREEN][j - 1]
            fw = "700" if j == 0 else "400"
            lines.append(f'<text x="{tip_x + 8:.0f}" y="{ty}" fill="{color}" font-size="10" font-weight="{fw}">{tl}</text>')
        lines.append('</g>')
        # Hover highlight column
        lines.append(
            f'<rect x="{cx:.1f}" y="{mt}" width="{col_w:.1f}" height="{ph}" '
            f'fill="white" opacity="0" cursor="crosshair" '
            f'onmouseover="document.getElementById(\'{gid}\').setAttribute(\'visibility\',\'visible\');this.setAttribute(\'opacity\',\'0.04\')" '
            f'onmouseout="document.getElementById(\'{gid}\').setAttribute(\'visibility\',\'hidden\');this.setAttribute(\'opacity\',\'0\')" '
            f'onclick="var g=document.getElementById(\'{gid}\');g.setAttribute(\'visibility\',g.getAttribute(\'visibility\')===\'visible\'?\'hidden\':\'visible\')"/>'
        )

    # X-axis date labels
    tick_interval = 3 if n <= 30 else 7
    for i in range(n):
        if i % tick_interval == 0:
            # Format: "Mar 18" from "2026-03-18"
            d = dates[i]
            month_names = {"01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May"}
            parts = d.split("-")
            label = f"{month_names.get(parts[1], parts[1])} {int(parts[2])}"
            lines.append(f'<text x="{x_pos(i):.1f}" y="{mt + ph + 16:.1f}" fill="{TEXT_DIM}" font-size="10" text-anchor="middle">{label}</text>')

    # Axis labels
    lines.append(f'<text x="{ml - 8}" y="{mt - 6}" fill="{TEXT_DIM}" font-size="10" text-anchor="end">Comments</text>')
    lines.append(f'<text x="{ml + pw + 8}" y="{mt - 6}" fill="{YELLOW}" font-size="10" text-anchor="start">PRs/day</text>')

    # Sweep annotation
    if sweep_start is not None:
        ax = x_pos(sweep_start)
        lines.append(f'<line x1="{ax:.1f}" x2="{ax:.1f}" y1="{mt}" y2="{mt + ph}" stroke="{ACCENT}" stroke-width="1.5" stroke-dasharray="4 3" opacity="0.6"/>')
        lines.append(f'<text x="{ax + 4:.1f}" y="{mt + 14}" fill="{ACCENT}" font-size="10">Agora sweep</text>')

    # Legend
    ly = h - 12
    lx = ml
    for label, color in [("Community comments", BLUE), ("Agora reviews", ACCENT), ("New PRs/day", YELLOW), ("PR updates/day", GREEN)]:
        lines.append(f'<rect x="{lx}" y="{ly - 8}" width="12" height="12" fill="{color}" rx="2" opacity="0.8"/>')
        lines.append(f'<text x="{lx + 16}" y="{ly + 2}" fill="{TEXT}" font-size="11">{label}</text>')
        lx += len(label) * 7 + 36

    lines.append('</svg>')
    return "".join(lines)


# ---------------------------------------------------------------------------
# Chart 5: Agora Review Impact (before/after comparison)
# ---------------------------------------------------------------------------

def generate_review_impact(csv_path: str = "") -> str:
    """Generate an SVG showing key before/after metrics around the Agora sweep."""
    import csv
    from pathlib import Path

    if not csv_path:
        csv_path = str(Path(__file__).parent.parent / "data" / "community_activity.csv")

    data_path = Path(csv_path)
    if not data_path.exists():
        return _empty_svg("No activity data available", 800, 280)

    rows = []
    with data_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["date"].strip():
                rows.append(row)

    if not rows:
        return _empty_svg("No activity data available", 800, 280)

    # Split into pre-sweep (before Apr 11) and sweep (Apr 11-12)
    pre_rows = [r for r in rows if r["date"] < "2026-04-11"]
    sweep_rows = [r for r in rows if r["date"] >= "2026-04-11" and r["date"] <= "2026-04-12"]

    if not pre_rows or not sweep_rows:
        return _empty_svg("Insufficient data for comparison", 800, 280)

    # Last 7 days before sweep
    pre_week = pre_rows[-7:] if len(pre_rows) >= 7 else pre_rows
    avg_comments_pre = sum(int(r["other_comments"]) for r in pre_week) / len(pre_week)
    avg_prs_pre = sum(int(r["prs_created"]) for r in pre_week) / len(pre_week)
    avg_updates_pre = sum(int(r["prs_updated"]) for r in pre_week) / len(pre_week)

    total_agora = sum(int(r["mato_comments"]) for r in sweep_rows)
    avg_comments_sweep = sum(int(r["other_comments"]) for r in sweep_rows) / len(sweep_rows)
    avg_prs_sweep = sum(int(r["prs_created"]) for r in sweep_rows) / len(sweep_rows)
    avg_updates_sweep = sum(int(r["prs_updated"]) for r in sweep_rows) / len(sweep_rows)

    w, h = 800, 280
    ml, mr, mt, mb = 40, 40, 50, 30

    lines = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Agora review impact" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{w}" height="{h}" rx="12" fill="{BG}" stroke="{BORDER}"/>',
        f'<text x="{w/2:.0f}" y="24" fill="{TEXT}" font-size="16" font-weight="700" text-anchor="middle">Agora Review Impact: Before vs. During Sweep</text>',
        f'<text x="{w/2:.0f}" y="40" fill="{TEXT_DIM}" font-size="11" text-anchor="middle">"Before" = 7-day avg (Apr 4-10). "During" = avg of Apr 11-12. {total_agora} Agora reviews posted.</text>',
    ]

    # Metric cards
    metrics = [
        ("Community Comments/day", avg_comments_pre, avg_comments_sweep),
        ("New PRs/day", avg_prs_pre, avg_prs_sweep),
        ("PR Updates/day", avg_updates_pre, avg_updates_sweep),
    ]

    card_w = (w - ml - mr - 40) / 3
    for i, (label, before, after) in enumerate(metrics):
        cx = ml + 20 + i * (card_w + 20) + card_w / 2
        cy = mt + 50

        # Card background
        card_x = ml + 20 + i * (card_w + 20)
        lines.append(f'<rect x="{card_x:.0f}" y="{mt + 20}" width="{card_w:.0f}" height="180" rx="8" fill="{BORDER}" opacity="0.5"/>')

        # Label
        lines.append(f'<text x="{cx:.0f}" y="{cy + 10}" fill="{TEXT}" font-size="12" font-weight="600" text-anchor="middle">{label}</text>')

        # Before value
        lines.append(f'<text x="{cx:.0f}" y="{cy + 45}" fill="{TEXT_DIM}" font-size="11" text-anchor="middle">Before</text>')
        lines.append(f'<text x="{cx:.0f}" y="{cy + 70}" fill="{BLUE}" font-size="28" font-weight="700" text-anchor="middle">{before:.0f}</text>')

        # After value
        lines.append(f'<text x="{cx:.0f}" y="{cy + 100}" fill="{TEXT_DIM}" font-size="11" text-anchor="middle">During</text>')

        # Color the "during" number based on increase/decrease
        change = after - before
        change_pct = (change / before * 100) if before > 0 else 0
        after_color = GREEN if change > 0 else RED
        lines.append(f'<text x="{cx:.0f}" y="{cy + 125}" fill="{after_color}" font-size="28" font-weight="700" text-anchor="middle">{after:.0f}</text>')

        # Change indicator
        arrow = "+" if change >= 0 else ""
        lines.append(f'<text x="{cx:.0f}" y="{cy + 148}" fill="{after_color}" font-size="13" font-weight="600" text-anchor="middle">{arrow}{change_pct:.0f}%</text>')

    lines.append('</svg>')
    return "".join(lines)


def _empty_svg(message: str, width: int, height: int) -> str:
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="{message}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="{width}" height="{height}" fill="{BG}" rx="12"/>'
        f'<text x="{width / 2:.0f}" y="{height / 2:.0f}" fill="{TEXT}" font-size="18" text-anchor="middle">{message}</text>'
        '</svg>'
    )
