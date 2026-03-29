#!/usr/bin/env python3
"""Generate the AGORA Phase 3 BPB timeline SVG."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from techniques import annotate_prs, is_neural_only, load_prs

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = REPO_ROOT / "data" / "pr_cache.json"

BG = "#161b22"
TEXT = "#e6edf3"
TEXT_DIM = "#8b949e"
GREEN = "#3fb950"
RED = "#f85149"
ACCENT = "#bc8cff"
BORDER = "#30363d"

PLOT_WIDTH = 1200
PLOT_HEIGHT = 520
MARGIN_LEFT = 92
MARGIN_RIGHT = 28
MARGIN_TOP = 44
MARGIN_BOTTOM = 72

EVENTS = (
    (date(2026, 3, 22), "GPTQ ruling", RED),
    (date(2026, 3, 24), "TTT ruling", GREEN),
    (date(2026, 3, 27), "N-gram massacre", RED),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_CACHE, help="Path to data/pr_cache.json")
    return parser.parse_args()


def _positive_bpb(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    if value <= 0:
        return None
    return float(value)


def _parse_created(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


def _timeline_points(prs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for pr in prs:
        created = _parse_created(pr.get("created"))
        bpb = _positive_bpb(pr.get("bpb"))
        if created is None or bpb is None:
            continue
        if pr.get("status") not in {"ALIVE", "AT_RISK", "DEAD"}:
            continue
        filtered.append({"created": created, "bpb": bpb, "number": pr.get("number"), "neural": is_neural_only(pr)})
    filtered.sort(key=lambda item: (item["created"], item["number"] or 0))
    return filtered


def _improvement_series(points: list[dict[str, Any]], *, neural_only: bool) -> list[dict[str, Any]]:
    best = float("inf")
    improvements: list[dict[str, Any]] = []
    for point in points:
        if neural_only and not point["neural"]:
            continue
        if point["bpb"] < best:
            best = point["bpb"]
            improvements.append(point)
    return improvements


def _format_bpb(value: float) -> str:
    if value >= 0.1:
        return f"{value:.3f}"
    if value >= 0.001:
        return f"{value:.4f}"
    return f"{value:.2e}"


def _x_scale(moment: datetime, start: datetime, end: datetime, plot_left: float, plot_width: float) -> float:
    span = (end - start).total_seconds() or 1.0
    return plot_left + (((moment - start).total_seconds() / span) * plot_width)


def _y_scale(value: float, log_min: float, log_max: float, plot_top: float, plot_height: float) -> float:
    span = (log_max - log_min) or 1.0
    return plot_top + (((math.log10(value) - log_min) / span) * plot_height)


def _step_path(series: list[dict[str, Any]], start: datetime, end: datetime, log_min: float, log_max: float) -> str:
    plot_left = MARGIN_LEFT
    plot_width = PLOT_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    plot_top = MARGIN_TOP
    plot_height = PLOT_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

    if not series:
        baseline_y = plot_top + plot_height
        return f"M {plot_left:.2f} {baseline_y:.2f} H {plot_left + plot_width:.2f}"

    first = series[0]
    first_x = _x_scale(first["created"], start, end, plot_left, plot_width)
    first_y = _y_scale(first["bpb"], log_min, log_max, plot_top, plot_height)
    commands = [f"M {plot_left:.2f} {first_y:.2f}", f"H {first_x:.2f}"]
    current_y = first_y
    for point in series[1:]:
        x = _x_scale(point["created"], start, end, plot_left, plot_width)
        y = _y_scale(point["bpb"], log_min, log_max, plot_top, plot_height)
        commands.append(f"H {x:.2f}")
        if abs(y - current_y) > 1e-6:
            commands.append(f"V {y:.2f}")
        current_y = y
    commands.append(f"H {plot_left + plot_width:.2f}")
    return " ".join(commands)


def _candidate_ticks(log_min: float, log_max: float) -> list[float]:
    ticks: list[float] = []
    for exponent in range(math.floor(log_min) - 1, math.ceil(log_max) + 2):
        for multiplier in (1.0, 3.0):
            value = multiplier * (10 ** exponent)
            log_value = math.log10(value)
            if log_min <= log_value <= log_max:
                ticks.append(value)
    return sorted(set(ticks), reverse=True)


def generate_svg(prs: list[dict[str, Any]]) -> str:
    points = _timeline_points(prs)
    if not points:
        return (
            f'<svg viewBox="0 0 {PLOT_WIDTH} {PLOT_HEIGHT}" width="100%" role="img" '
            f'aria-label="No BPB timeline data available" xmlns="http://www.w3.org/2000/svg">'
            f'<rect width="{PLOT_WIDTH}" height="{PLOT_HEIGHT}" fill="{BG}" rx="16"/>'
            f'<text x="{PLOT_WIDTH / 2:.0f}" y="{PLOT_HEIGHT / 2:.0f}" fill="{TEXT}" font-size="22" text-anchor="middle">No timeline data available</text>'
            '</svg>'
        )

    all_series = _improvement_series(points, neural_only=False)
    neural_series = _improvement_series(points, neural_only=True)
    point_start = datetime.combine(points[0]["created"].date(), datetime.min.time())
    point_end = datetime.combine(points[-1]["created"].date(), datetime.max.time())
    start = min(point_start, datetime.combine(EVENTS[0][0], datetime.min.time()))
    end = max(point_end, datetime.combine(EVENTS[-1][0], datetime.max.time()))

    values = [point["bpb"] for point in all_series] + [point["bpb"] for point in neural_series]
    log_min = math.log10(min(values)) - 0.12
    log_max = math.log10(max(values)) + 0.18

    plot_left = MARGIN_LEFT
    plot_width = PLOT_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    plot_top = MARGIN_TOP
    plot_height = PLOT_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    plot_bottom = plot_top + plot_height
    plot_right = plot_left + plot_width

    y_ticks = _candidate_ticks(log_min, log_max)
    x_ticks: list[datetime] = []
    current_day = start.date()
    final_day = end.date()
    while current_day <= final_day:
        if (current_day - start.date()).days % 2 == 0:
            x_ticks.append(datetime.combine(current_day, datetime.min.time()))
        current_day += timedelta(days=1)
    x_ticks.append(datetime.combine(final_day, datetime.min.time()))
    seen = set()
    x_ticks = [tick for tick in x_ticks if not (tick in seen or seen.add(tick))]

    lines: list[str] = [
        f'<svg viewBox="0 0 {PLOT_WIDTH} {PLOT_HEIGHT}" width="100%" role="img" aria-label="BPB timeline" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{PLOT_WIDTH}" height="{PLOT_HEIGHT}" rx="16" fill="{BG}" stroke="{BORDER}"/>',
        f'<text x="{plot_left}" y="26" fill="{TEXT}" font-size="22" font-weight="700">Cumulative SOTA BPB by PR creation date</text>',
        f'<text x="{plot_left}" y="44" fill="{TEXT_DIM}" font-size="13">Log scale. Neural-only strips cache-heavy rulings; all-techniques keeps every classified result with a final status.</text>',
    ]

    for value in y_ticks:
        y = _y_scale(value, log_min, log_max, plot_top, plot_height)
        lines.append(f'<line x1="{plot_left}" x2="{plot_right}" y1="{y:.2f}" y2="{y:.2f}" stroke="{BORDER}" stroke-width="1"/>')
        lines.append(f'<text x="{plot_left - 12}" y="{y + 4:.2f}" fill="{TEXT_DIM}" font-size="12" text-anchor="end">{_format_bpb(value)}</text>')

    for tick in x_ticks:
        x = _x_scale(tick, start, end, plot_left, plot_width)
        lines.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{plot_top}" y2="{plot_bottom}" stroke="{BORDER}" stroke-width="1" opacity="0.55"/>')
        lines.append(f'<text x="{x:.2f}" y="{plot_bottom + 24:.2f}" fill="{TEXT_DIM}" font-size="12" text-anchor="middle">{tick.strftime("Mar %-d")}</text>')

    for event_day, label, color in EVENTS:
        event_x = _x_scale(datetime.combine(event_day, datetime.min.time()), start, end, plot_left, plot_width)
        lines.append(f'<line x1="{event_x:.2f}" x2="{event_x:.2f}" y1="{plot_top}" y2="{plot_bottom}" stroke="{color}" stroke-width="2" stroke-dasharray="6 6" opacity="0.8"/>')
        lines.append(f'<text x="{event_x + 6:.2f}" y="{plot_top + 14:.2f}" fill="{color}" font-size="12">{label}</text>')

    lines.extend(
        [
            f'<rect x="{plot_left}" y="{plot_top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="{BORDER}"/>',
            f'<path d="{_step_path(all_series, start, end, log_min, log_max)}" fill="none" stroke="{ACCENT}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>',
            f'<path d="{_step_path(neural_series, start, end, log_min, log_max)}" fill="none" stroke="{GREEN}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>',
        ]
    )

    for color, series, label in ((ACCENT, all_series, "All-techniques"), (GREEN, neural_series, "Neural-only")):
        for point in series:
            x = _x_scale(point["created"], start, end, plot_left, plot_width)
            y = _y_scale(point["bpb"], log_min, log_max, plot_top, plot_height)
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}" stroke="{BG}" stroke-width="1.5"/>')
        final_point = series[-1]
        final_x = _x_scale(final_point["created"], start, end, plot_left, plot_width)
        final_y = _y_scale(final_point["bpb"], log_min, log_max, plot_top, plot_height)
        label_x = min(plot_right - 6, final_x + 10)
        lines.append(f'<text x="{label_x:.2f}" y="{final_y - 10:.2f}" fill="{color}" font-size="12">{label}: {_format_bpb(final_point["bpb"])} BPB</text>')

    legend_y = PLOT_HEIGHT - 24
    mid_y = plot_top + (plot_height / 2)
    lines.extend(
        [
            f'<line x1="{plot_left}" x2="{plot_left + 26}" y1="{legend_y:.2f}" y2="{legend_y:.2f}" stroke="{GREEN}" stroke-width="3"/>',
            f'<text x="{plot_left + 34}" y="{legend_y + 4:.2f}" fill="{TEXT}" font-size="13">Neural-only</text>',
            f'<line x1="{plot_left + 150}" x2="{plot_left + 176}" y1="{legend_y:.2f}" y2="{legend_y:.2f}" stroke="{ACCENT}" stroke-width="3"/>',
            f'<text x="{plot_left + 184}" y="{legend_y + 4:.2f}" fill="{TEXT}" font-size="13">All-techniques</text>',
            f'<text x="22" y="{mid_y:.2f}" fill="{TEXT_DIM}" font-size="13" transform="rotate(-90 22 {mid_y:.2f})">BPB (log scale)</text>',
            f'<text x="{plot_left + (plot_width / 2):.2f}" y="{PLOT_HEIGHT - 8:.2f}" fill="{TEXT_DIM}" font-size="13" text-anchor="middle">Date created</text>',
            '</svg>',
        ]
    )
    return "".join(lines)


def main() -> int:
    args = parse_args()
    prs = annotate_prs(load_prs(args.input))
    print(generate_svg(prs))
    print(f"[timeline] generated SVG from {len(prs)} PRs", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
