#!/usr/bin/env python3
"""
run_pipeline.py — Orchestrator for the AGORA site pipeline.

Steps:
  1. Run fetch_prs.py to update data/pr_cache.json
  2. Run fetch_community.py to sync community issue-form submissions
  3. Run classify.py to classify all PRs
  4. Run techniques.py to build data/techniques.json
  5. Run timeline.py to generate inline SVG
  6. Read index.html
  7. Replace leaderboard / techniques / timeline sentinel sections
  8. Update the version bar timestamp
  9. Write updated index.html

The HTML replacement uses sentinel comment blocks that are injected on first run
and stable on subsequent runs:
    <!-- AGORA:LEADERBOARD_NEURAL_START -->
    ...rows...
    <!-- AGORA:LEADERBOARD_NEURAL_END -->
and:
    <!-- AGORA:LEADERBOARD_ARCHIVE_START -->
    ...rows...
    <!-- AGORA:LEADERBOARD_ARCHIVE_END -->

Usage:
    python scripts/run_pipeline.py

Environment:
    GITHUB_TOKEN — used by fetch_prs.py and required for fetch_community.py
"""

import json
import os
import re
import subprocess
import sys
from html import escape
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
CACHE_PATH = REPO_ROOT / "data" / "pr_cache.json"
TECHNIQUES_PATH = REPO_ROOT / "data" / "techniques.json"
CHANGELOG_PATH = REPO_ROOT / "data" / "changelog.json"
INDEX_HTML_PATH = REPO_ROOT / "index.html"
SCRIPTS_DIR = Path(__file__).parent

NEURAL_START = "<!-- AGORA:LEADERBOARD_NEURAL_START -->"
NEURAL_END = "<!-- AGORA:LEADERBOARD_NEURAL_END -->"
ARCHIVE_START = "<!-- AGORA:LEADERBOARD_ARCHIVE_START -->"
ARCHIVE_END = "<!-- AGORA:LEADERBOARD_ARCHIVE_END -->"
TECHNIQUES_START = "<!-- AGORA:TECHNIQUES_START -->"
TECHNIQUES_END = "<!-- AGORA:TECHNIQUES_END -->"
TIMELINE_START = "<!-- AGORA:TIMELINE_START -->"
TIMELINE_END = "<!-- AGORA:TIMELINE_END -->"
CHANGELOG_START = "<!-- AGORA:CHANGELOG_START -->"
CHANGELOG_END = "<!-- AGORA:CHANGELOG_END -->"
CHECKLIST_START = "<!-- AGORA:CHECKLIST_START -->"
CHECKLIST_END = "<!-- AGORA:CHECKLIST_END -->"
RULINGS_START = "<!-- AGORA:RULINGS_START -->"
RULINGS_END = "<!-- AGORA:RULINGS_END -->"
ALERTS_START = "<!-- AGORA:ALERTS_START -->"
ALERTS_END = "<!-- AGORA:ALERTS_END -->"
COMPUTE_START = "<!-- AGORA:COMPUTE_START -->"
COMPUTE_END = "<!-- AGORA:COMPUTE_END -->"

CHECKLIST_PATH = REPO_ROOT / "data" / "checklist.json"
RULINGS_PATH = REPO_ROOT / "data" / "rulings.json"
ALERTS_PATH = REPO_ROOT / "data" / "alerts.json"
COMPUTE_PATH = REPO_ROOT / "data" / "compute_guide.json"

PR_BASE_URL = "https://github.com/openai/parameter-golf/pull"


# ---------------------------------------------------------------------------
# Step 1 & 2: Run sub-scripts
# ---------------------------------------------------------------------------

def _run_script(script_name: str, *args: str, capture_stdout: bool = False) -> bool | str:
    """Run a script in the scripts/ directory."""
    script_path = SCRIPTS_DIR / script_name
    env = dict(os.environ)
    cmd = [sys.executable, str(script_path), *args]
    print(f"[PIPELINE] Running {' '.join([script_name, *args]).strip()}...", flush=True)
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=capture_stdout,
        text=capture_stdout,
    )
    if capture_stdout and result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        print(f"[PIPELINE] {script_name} exited with code {result.returncode}", flush=True)
        return False
    if capture_stdout:
        return result.stdout
    return True


# ---------------------------------------------------------------------------
# HTML generation helpers
# ---------------------------------------------------------------------------

def _format_bpb(bpb: float | None) -> str:
    """Format BPB value for table display."""
    if bpb is None:
        return "?"
    return f"{bpb:.4f}"


def _format_seeds(seeds: int | None) -> str:
    """Format seed count as 'N/3' or '?'."""
    if seeds is None:
        return "?"
    return f"{min(seeds, 3)}/3"


def _format_artifact(artifact_bytes: int | None) -> str:
    """Format artifact size as 'XX.X MB' or '?'."""
    if artifact_bytes is None:
        return "?"
    mb = artifact_bytes / 1_000_000
    return f"{mb:.1f} MB"


def _status_badge(status: str, seeds: int | None) -> str:
    """Render an HTML badge for a PR status with tooltip."""
    badges: dict[str, tuple[str, str, str]] = {
        "ALIVE": ("badge-alive", "ALIVE", "Open PR, 3/3 seeds, known artifact, passes compliance"),
        "DEAD": ("badge-dead", "DEAD", "Closed by maintainer for compliance violation"),
        "NOTABLE": ("badge-notable", "NOTABLE", "Accepted as Notable Non-Record by competition organizers"),
        "AT_RISK": ("badge-risk", "AT-RISK", "Open but flagged — banned technique keywords detected"),
        "INCOMPLETE": ("badge-incomplete", "INCOMPLETE", "Missing required fields (seeds, artifact, or BPB)"),
        "UNKNOWN": ("badge-incomplete", "UNKNOWN", "Classification failed — check flags"),
    }
    if status == "INCOMPLETE" and seeds is not None and seeds < 3:
        cls = "badge-incomplete"
        label = f"{seeds} SEED{'S' if seeds != 1 else ''}"
        tip = f"Only {seeds}/3 seeds submitted — needs all 3 (42, 1337, 2024)"
    else:
        cls, label, tip = badges.get(status, ("badge-incomplete", status, ""))
    return f'<span class="badge {cls}"{_tooltip_attr(tip)}>{label}</span>'


def _type_badge(technique_type: str) -> str:
    """Render an HTML badge for technique type with tooltip."""
    badges: dict[str, tuple[str, str, str]] = {
        "neural": ("badge-neural", "Neural", "Pure neural architecture — no caching or test-time training"),
        "cache": ("badge-cache", "Cache", "Uses n-gram or eval-time caching — check compliance"),
        "ttt": ("badge-ttt", "TTT", "Test-time training — adapts model weights during evaluation"),
        "hybrid": ("badge-hybrid", "Hybrid", "Combines neural + cache or TTT techniques"),
    }
    cls, label, tip = badges.get(technique_type, ("badge-incomplete", technique_type or "?", "Type unknown"))
    return f'<span class="badge {cls}"{_tooltip_attr(tip)}>{label}</span>'


def _track_badge(track: str) -> str:
    """Render an HTML badge for submission track with tooltip."""
    badges: dict[str, tuple[str, str, str]] = {
        "record": ("badge-record", "Record", "Main track: 10min on 8xH100 SXM, artifact ≤ 16MB"),
        "non-record": ("badge-nonrecord", "Non-Record", "Notable submission outside record constraints"),
    }
    cls, label, tip = badges.get(track, ("badge-incomplete", track or "?", "Track unknown"))
    return f'<span class="badge {cls}"{_tooltip_attr(tip)}>{label}</span>'


def _pr_link(number: int) -> str:
    """Render a PR number as an HTML link."""
    return f'<a href="{PR_BASE_URL}/{number}">#{number}</a>'


def _technique_badge(kind: str, count: int) -> str:
    if kind == "banned":
        return f'<span class="badge badge-banned"{_tooltip_attr("Banned by maintainers — submissions using this are DEAD")}>BANNED</span>'
    if count == 0:
        return f'<span class="badge badge-grey"{_tooltip_attr("No PRs have tried this technique yet")}>UNTRIED</span>'
    return f'<span class="badge badge-legal"{_tooltip_attr("Legal technique — allowed in submissions")}>LEGAL</span>'


def _render_technique_card(technique: dict[str, Any]) -> str:
    best_alive = technique.get("best_alive")
    best_html = "No ALIVE run yet"
    if best_alive:
        best_html = f"{_pr_link(int(best_alive['number']))} &middot; {_format_bpb(best_alive.get('bpb'))}"

    modifier = ""
    if technique.get("kind") == "banned":
        modifier += " technique-card-banned"
    if technique.get("count", 0) == 0:
        modifier += " technique-card-empty"

    return (
        f'<article class="technique-card{modifier}">'
        f'<div class="technique-card-head">'
        f'<h3>{escape(technique["name"])}</h3>'
        f'{_technique_badge(technique.get("kind", "legal"), int(technique.get("count", 0)))}'
        f'</div>'
        f'<p class="technique-card-sub">{technique.get("count", 0)} PRs &middot; {technique.get("author_count", 0)} people</p>'
        f'<div class="technique-metrics">'
        f'<div><span>Best ALIVE</span><strong>{best_html}</strong></div>'
        f'<div><span>Alive / Dead</span><strong>{technique.get("alive_count", 0)} / {technique.get("dead_count", 0)}</strong></div>'
        f'</div>'
        f'<p class="technique-card-foot">AT-RISK {technique.get("at_risk_count", 0)} &middot; INCOMPLETE {technique.get("incomplete_count", 0)}</p>'
        f'</article>'
    )


def _render_collaboration_line(technique: dict[str, Any]) -> str:
    people: list[dict[str, Any]] = technique.get("people", [])
    visible_people = people[:6]
    chunks: list[str] = []
    for person in visible_people:
        pr_links = ", ".join(
            _pr_link(int(pr["number"]))
            for pr in person.get("prs", [])[:4]
            if pr.get("number") is not None
        )
        if len(person.get("prs", [])) > 4:
            pr_links += f" +{len(person['prs']) - 4}"
        chunks.append(f"@{escape(person['author'])} ({pr_links})")

    overflow = technique.get("author_count", 0) - len(visible_people)
    overflow_html = f' <span class="technique-overflow">+{overflow} more people</span>' if overflow > 0 else ""
    return (
        f'<div class="collab-item">'
        f'<div class="collab-title">'
        f'<strong>{technique.get("author_count", 0)} people working on {escape(technique["name"])}</strong>'
        f'<span class="badge badge-grey">{technique.get("count", 0)} PRs</span>'
        f'</div>'
        f'<div class="collab-links">{"; ".join(chunks)}{overflow_html}</div>'
        f'</div>'
    )


def _render_techniques_section(techniques_payload: dict[str, Any]) -> str:
    techniques = techniques_payload.get("techniques", [])
    collaboration = [
        item for item in techniques_payload.get("collaboration", []) if item.get("author_count", 0) >= 2
    ][:10]
    combinations = techniques_payload.get("untried_combinations", [])[:8]
    untouched = techniques_payload.get("summary", {}).get("untouched_techniques", [])

    intro_bits = [
        f"Scanned {techniques_payload.get('summary', {}).get('pr_count', 0)} classified PRs.",
        f"{len(techniques) - len(untouched)}/{len(techniques)} named techniques have at least one hit.",
    ]
    if untouched:
        intro_bits.append("Untouched: " + ", ".join(escape(name) for name in untouched) + ".")

    cards_html = "\n".join(_render_technique_card(item) for item in techniques)

    if collaboration:
        collaboration_html = "\n".join(_render_collaboration_line(item) for item in collaboration)
    else:
        collaboration_html = '<p style="color:var(--text-dim);">No multi-person clusters detected yet.</p>'

    if combinations:
        combinations_html = "".join(
            (
                '<div class="combo-chip">'
                f"<strong>{escape(item['left'])} &times; {escape(item['right'])}</strong>"
                f'<span>{item["left_count"]} + {item["right_count"]} runs, zero combined PRs</span>'
                '</div>'
            )
            for item in combinations
        )
    else:
        combinations_html = '<p style="color:var(--text-dim);">No clean untried combinations surfaced yet.</p>'

    return (
        f'<p style="color:var(--text-dim);margin-bottom:1rem;">{" ".join(intro_bits)}</p>'
        '<div class="card">'
        '<h3 style="margin-top:0;">Technique Map</h3>'
        f'<div class="technique-grid">{cards_html}</div>'
        '</div>'
        '<div class="card">'
        '<h3 style="margin-top:0;">Collaboration Finder</h3>'
        '<p style="color:var(--text-dim);margin-bottom:1rem;">Top clusters where multiple researchers are circling the same idea.</p>'
        f'<div class="collab-list">{collaboration_html}</div>'
        '</div>'
        '<div class="card">'
        '<h3 style="margin-top:0;">Untried Combinations</h3>'
        '<p style="color:var(--text-dim);margin-bottom:1rem;">Popular pairings with zero combined PRs so far.</p>'
        f'<div class="combo-grid">{combinations_html}</div>'
        '</div>'
    )


def _render_timeline_section(svg: str, prs: list[dict[str, Any]]) -> str:
    tracked = sum(
        1
        for pr in prs
        if pr.get("bpb") not in (None, 0, 0.0) and pr.get("status") in {"ALIVE", "AT_RISK", "DEAD"}
    )
    toggle_js = (
        '<script>'
        'function toggleSeries(id,btn){'
        'var g=document.getElementById(id);'
        'if(!g)return;'
        'var vis=g.style.display!=="none";'
        'g.style.display=vis?"none":"";'
        'btn.style.opacity=vis?"0.4":"1";'
        '}'
        '</script>'
    )
    toggle_buttons = (
        '<div style="display:flex;gap:0.75rem;margin-bottom:0.75rem;">'
        '<button onclick="toggleSeries(\'series-neural\',this)" '
        'style="background:var(--surface);border:1px solid #3fb950;color:#3fb950;padding:0.3rem 0.8rem;'
        'border-radius:6px;cursor:pointer;font-size:0.85rem;">Neural-only</button>'
        '<button onclick="toggleSeries(\'series-all\',this)" '
        'style="background:var(--surface);border:1px solid #bc8cff;color:#bc8cff;padding:0.3rem 0.8rem;'
        'border-radius:6px;cursor:pointer;font-size:0.85rem;">All-techniques</button>'
        '</div>'
    )
    return (
        '<p style="color:var(--text-dim);margin-bottom:1rem;">'
        f'Cumulative SOTA curve built from {tracked} classified PRs with a positive BPB and final status '
        'ALIVE, AT-RISK, or DEAD. Neural-only strips cache-heavy rulings so the March 27 cliff stays visible.'
        '</p>'
        f'{toggle_buttons}'
        '<div class="card chart-card">'
        f'{svg}'
        '</div>'
        f'{toggle_js}'
    )


def _render_with_tooltip(text: str, tooltip: str | None) -> str:
    """Render text wrapped in a <span> with a tooltip title attribute.

    If tooltip is None or empty, returns text unwrapped.
    """
    if not tooltip:
        return text
    return f'<span title="{escape(tooltip)}">{text}</span>'


def _tooltip_attr(tooltip: str | None) -> str:
    """Return a title='...' attribute string, or empty string if no tooltip."""
    if not tooltip:
        return ""
    return f' title="{escape(tooltip)}"'


def _render_checklist(data: dict[str, Any]) -> str:
    """Render the submission checklist + technique legality from data/checklist.json."""
    updated = escape(data.get("last_updated", ""))
    items = data.get("items", [])
    techniques = data.get("techniques", {})

    # Checklist card
    checklist_lis = "\n".join(
        f'  <li{_tooltip_attr(item.get("tooltip"))}>{item["text"]}</li>'
        for item in items
    )
    html = (
        f'<p style="color:var(--text-dim);margin-bottom:1rem;">Updated {updated}. Check ALL boxes before submitting.</p>\n'
        '<div class="card">\n'
        '<h3>Before You Submit</h3>\n'
        f'<ul class="checklist">\n{checklist_lis}\n</ul>\n'
        '</div>\n'
    )

    # Technique legality card
    legal_lis = "\n".join(
        f'  <li class="legal"{_tooltip_attr(t.get("tooltip"))}>{t["text"]}</li>'
        for t in techniques.get("legal", [])
    )
    banned_lis = "\n".join(
        f'  <li class="banned"{_tooltip_attr(t.get("tooltip"))}>{t["text"]}</li>'
        for t in techniques.get("banned", [])
    )
    grey_lis = "\n".join(
        f'  <li class="grey"{_tooltip_attr(t.get("tooltip"))}>{t["text"]}</li>'
        for t in techniques.get("grey_area", [])
    )
    html += (
        '<div class="card">\n'
        '<h3>Techniques &mdash; What\'s Legal RIGHT NOW</h3>\n'
        f'<ul class="technique-list">\n{legal_lis}\n</ul>\n'
        f'<ul class="technique-list" style="margin-top:1rem;">\n{banned_lis}\n</ul>\n'
        f'<ul class="technique-list" style="margin-top:1rem;">\n{grey_lis}\n</ul>\n'
        '</div>'
    )
    return html


def _render_rulings(rulings: list[dict[str, Any]]) -> str:
    """Render the rule change history from data/rulings.json."""
    entries: list[str] = []
    for ruling in rulings:
        date = escape(ruling.get("date", ""))
        desc = escape(ruling.get("description", ""))
        rtype = ruling.get("type", "ban")
        modifier = " legal-change" if rtype == "clarification" else ""

        parts = [
            f'<div class="timeline-entry{modifier}">',
            f'  <div class="timeline-date">{date}</div>',
            f'  <p>{desc}</p>',
        ]

        affected = ruling.get("affected_prs", [])
        if affected:
            pr_list = ", ".join(f"#{n}" for n in affected)
            parts.append(
                f'  <p style="color:var(--text-dim);font-size:0.85rem;margin-top:0.25rem;">Affected: {pr_list}</p>'
            )

        source = ruling.get("source")
        source_label = ruling.get("source_label", "")
        if source:
            parts.append(
                f'  <p style="color:var(--text-dim);font-size:0.85rem;">Source: <a href="{escape(source)}">{escape(source_label)}</a></p>'
            )
        elif source_label:
            parts.append(
                f'  <p style="color:var(--text-dim);font-size:0.85rem;">Source: {escape(source_label)}</p>'
            )

        parts.append('</div>')
        entries.append("\n".join(parts))

    return (
        '<div class="card">\n'
        '<h3>Rule Change History</h3>\n'
        + "\n".join(entries) +
        '\n</div>'
    )


def _render_alerts(alerts: list[dict[str, Any]]) -> str:
    """Render community bug alerts from data/alerts.json."""
    if not alerts:
        return ""

    items: list[str] = []
    for i, alert in enumerate(alerts):
        title = alert.get("title", "")
        severity = alert.get("severity", "warning")
        badge_text = escape(alert.get("badge_text", ""))
        desc = alert.get("description", "")  # may contain HTML tags like <code>
        source = alert.get("source", "")
        source_label = escape(alert.get("source_label", ""))

        color = "var(--red)" if severity == "critical" else "var(--yellow)"
        badge_cls = "badge-banned" if severity == "critical" else "badge-grey"

        margin = ' style="margin-bottom:1rem;"' if i < len(alerts) - 1 else ""
        source_html = ""
        if source:
            source_html = f'\n<p style="font-size:0.8rem;color:var(--text-dim);">Source: <a href="{escape(source)}">{source_label}</a></p>'

        items.append(
            f'<div{margin}>\n'
            f'<strong style="color:{color};">{escape(title)}</strong> '
            f'<span class="badge {badge_cls}">{badge_text}</span>\n'
            f'<p style="font-size:0.85rem;color:var(--text-dim);margin-top:0.25rem;">{desc}</p>'
            f'{source_html}\n'
            '</div>'
        )

    return (
        '<div class="card" style="border-color:var(--yellow);border-width:2px;">\n'
        '<h3 style="margin-top:0;color:var(--yellow);">&#9888; Community Bug Alerts</h3>\n'
        '<p style="font-size:0.9rem;margin-bottom:0.75rem;">Known issues that may affect your BPB score. Check before submitting.</p>\n'
        + "\n".join(items) +
        '\n</div>'
    )


def _render_compute_guide(data: dict[str, Any]) -> str:
    """Render the compute survival guide from data/compute_guide.json."""
    golden = data.get("golden_rule", {})
    providers = data.get("providers", [])
    smoke_tests = data.get("smoke_tests", [])
    hw = data.get("hardware_comparison", {})

    # Golden rule card
    comparison = golden.get("comparison", {})
    bad = comparison.get("bad", {})
    good = comparison.get("good", {})
    bad_lis = "\n".join(f"      <li>{escape(item)}</li>" for item in bad.get("items", []))
    good_lis = "\n".join(f"      <li>{escape(item)}</li>" for item in good.get("items", []))

    html = (
        f'<div class="card" style="border-color:var(--red);">\n'
        f'<h3 style="color:var(--red);">{escape(golden.get("title", ""))}</h3>\n'
        f'<p>{escape(golden.get("description", ""))} <strong style="color:var(--yellow);">{golden.get("highlight", "")}</strong> &mdash;{golden.get("description_suffix", "")}</p>\n'
        '<div class="guide-compare">\n'
        f'  <div class="bad">\n    <h4>{escape(bad.get("heading", ""))}</h4>\n    <ul>\n{bad_lis}\n    </ul>\n  </div>\n'
        f'  <div class="good">\n    <h4>{escape(good.get("heading", ""))}</h4>\n    <ul>\n{good_lis}\n    </ul>\n  </div>\n'
        '</div>\n'
        '</div>\n'
    )

    # Provider cards
    for provider in providers:
        name = escape(provider.get("name", ""))
        tag = provider.get("tag", "")
        tag_str = f" ({escape(tag)})" if tag else ""
        desc = provider.get("description", "")  # may contain <code>
        docker = provider.get("docker_image")
        note = provider.get("note")
        tips = provider.get("tips")

        html += f'<div class="card">\n<h3>{name}{tag_str}</h3>\n'
        if docker:
            html += f'<p>Docker image: <code>{escape(docker)}</code></p>\n'
        html += f'<p>{desc}</p>\n'
        if tips:
            html += f'<p>{tips}</p>\n'
        if note:
            html += f'<p style="color:var(--text-dim);font-size:0.85rem;">{note}</p>\n'
        html += '</div>\n'

    # Smoke tests card
    def _smoke_test_cmd(t: dict[str, Any]) -> str:
        cmd = f'<code>{escape(t["command"])}</code>'
        note = t.get("command_note")
        if note:
            cmd += f" ({escape(note)})"
        return cmd

    test_rows = "\n".join(
        f'<tr><td>{escape(t["name"])}</td><td>{escape(t["catches"])}</td><td>{_smoke_test_cmd(t)}</td></tr>'
        for t in smoke_tests
    )
    html += (
        '<div class="card">\n'
        '<h3>CPU Smoke Tests (FREE &mdash; do these FIRST)</h3>\n'
        '<p>We\'ve run 5+ smoke tests during competition development. Each catches different failure modes:</p>\n'
        '<table style="margin:0.5rem 0;">\n'
        '<thead><tr><th>Test</th><th>What it catches</th><th>Command</th></tr></thead>\n'
        f'<tbody>\n{test_rows}\n</tbody>\n'
        '</table>\n'
        '<p style="color:var(--text-dim);font-size:0.85rem;">If it doesn\'t work on CPU, it won\'t work on GPU. Save your money. '
        'Our <code>cpu_test.py</code> is in the <a href="https://github.com/MatoTeziTanka/parameter-golf-private">community toolkit</a>.</p>\n'
        '</div>\n'
    )

    # Hardware comparison card
    hw_rows = "\n".join(
        f'<tr><td>{escape(r["gpu"])}</td><td>{escape(r["nvlink"])}</td><td>{escape(r["ddp_speed"])}</td><td>{escape(r["where"])}</td></tr>'
        for r in hw.get("rows", [])
    )
    hw_note = hw.get("note", "")
    html += (
        '<div class="card">\n'
        f'<h3>{escape(hw.get("title", "H100 SXM vs PCIe"))}</h3>\n'
        '<table>\n'
        '<thead><tr><th>GPU</th><th>NVLink</th><th>DDP Speed</th><th>Where</th></tr></thead>\n'
        f'<tbody>\n{hw_rows}\n</tbody>\n'
        '</table>\n'
        f'<p style="color:var(--text-dim);font-size:0.85rem;margin-top:0.5rem;">{escape(hw_note)}</p>\n'
        '</div>'
    )

    return html


def _render_changelog(changelog: list[dict[str, Any]]) -> str:
    """Render changelog entries from data/changelog.json into HTML cards."""
    roadmap_items = [
        ("v0.6.0", "Cited research tracker (papers &rarr; PRs &rarr; BPB)"),
        ("v0.7.0", "Cost efficiency ranking (dollars per BPB point)"),
        ("v0.8.0", "Review queue metrics (PR wait times, peer reviewer recognition)"),
        ("v1.0.0", "Community governance (threshold-based classification disputes)"),
    ]
    cards: list[str] = []
    for entry in changelog:
        version = escape(entry["version"])
        date = escape(entry["date"])
        title = escape(entry["title"])
        items = "".join(f"  <li>{escape(c)}</li>\n" for c in entry.get("changes", []))
        cards.append(
            f'<div class="card">\n'
            f'<h3 style="margin-top:0;color:var(--green);">v{version} &mdash; {date} ({title})</h3>\n'
            f'<ul style="font-size:0.85rem;color:var(--text-dim);list-style:disc;padding-left:1.5rem;">\n'
            f'{items}</ul>\n'
            f'</div>'
        )
    # Roadmap
    roadmap_lis = "".join(
        f'  <li>&#9744; <strong>{v}</strong> &mdash; {desc}</li>\n'
        for v, desc in roadmap_items
    )
    cards.append(
        '<div class="card" style="border-color:var(--border);opacity:0.7;">\n'
        '<h3 style="margin-top:0;color:var(--text-dim);">Roadmap</h3>\n'
        '<ul style="font-size:0.85rem;color:var(--text-dim);list-style:none;padding-left:0;">\n'
        f'{roadmap_lis}</ul>\n'
        '</div>'
    )
    return "\n".join(cards)


def _update_version_bar_from_changelog(html: str, changelog: list[dict[str, Any]], now: datetime) -> str:
    """Sync the version bar with the latest changelog entry."""
    if not changelog:
        return html
    latest = changelog[0]
    version = latest["version"]
    title = latest["title"]
    date_str = now.strftime("%B %-d, %Y")
    # Replace version
    html = re.sub(
        r'<strong style="color:var\(--accent\);">v[^<]+</strong>',
        f'<strong style="color:var(--accent);">v{escape(version)}</strong>',
        html,
        count=1,
    )
    # Replace date and subtitle
    html = re.sub(
        r"(Last updated:\s*)[^&<]+(&middot;)\s*[^&<]+(&middot;)",
        rf"\g<1>{date_str} \g<2> {escape(title)} \g<3>",
        html,
        count=1,
    )
    return html


def _dead_reason(pr: dict[str, Any]) -> str:
    """Extract a short dead reason from maintainer comments or flags."""
    flags: list[str] = pr.get("flags", [])
    for flag in flags:
        if "n-gram" in flag or "ngram" in flag:
            return "N-gram cache (03-27)"
        if "two-pass" in flag or "two_pass" in flag:
            return "Two-pass (03-27)"
        if "gptq" in flag:
            return "GPTQ on eval tokens (03-27)"
        if "prefill" in flag:
            return "Prefill cache (03-27)"
    # Try to extract from maintainer comment
    for comment in pr.get("maintainer_comments", []):
        comment_lower = comment.lower()
        if "n-gram" in comment_lower or "ngram" in comment_lower:
            return "N-gram cache (03-27)"
        if "two-pass" in comment_lower or "two_pass" in comment_lower:
            return "Two-pass (03-27)"
        if "gptq" in comment_lower:
            return "GPTQ on eval tokens (03-27)"
    return pr.get("technique_summary") or "Compliance violation"


# ---------------------------------------------------------------------------
# Leaderboard table row builders
# ---------------------------------------------------------------------------

def _build_row(rank: int, pr: dict[str, Any], include_reason: bool = False) -> str:
    """Build a single table row with data attributes for filtering.

    Column order: Status | # | PR | Author | BPB | Seeds | Artifact | Type | Track [| Reason]
    """
    num = pr["number"]
    author = pr.get("author", "unknown")
    bpb = _format_bpb(pr.get("bpb"))
    seeds = _format_seeds(pr.get("seeds"))
    artifact = _format_artifact(pr.get("artifact_bytes"))
    status = pr.get("status", "UNKNOWN")
    status_badge = _status_badge(status, pr.get("seeds"))
    ttype = pr.get("technique_type", "unknown")
    track = pr.get("track", "unknown")

    attrs = f'data-status="{status.lower()}" data-type="{ttype}" data-track="{track}"'

    cells = (
        f"<td>{status_badge}</td>"
        f"<td>{rank}</td>"
        f"<td>{_pr_link(num)}</td>"
        f"<td>@{author}</td>"
        f"<td>{bpb}</td>"
        f"<td>{seeds}</td>"
        f"<td>{artifact}</td>"
        f"<td>{_type_badge(ttype)}</td>"
        f"<td>{_track_badge(track)}</td>"
    )
    if include_reason:
        reason = "&mdash;" if status == "ALIVE" else _dead_reason(pr)
        cells += f"<td>{reason}</td>"

    return f"<tr {attrs}>{cells}</tr>"


def _build_neural_rows(prs: list[dict[str, Any]]) -> str:
    """Build tbody rows for the Neural-Only leaderboard.

    Shows ALIVE PRs only, with 3/3 seeds, known artifact, BPB >= 0.5.
    Sorted by BPB ascending (lower = better).
    """
    alive = [
        p for p in prs
        if p.get("status") == "ALIVE"
        and p.get("bpb") is not None
        and p.get("bpb", 0) >= 0.5
        and p.get("seeds") is not None
        and p.get("seeds", 0) >= 3
        and p.get("artifact_bytes") is not None
    ]
    alive.sort(key=lambda p: p["bpb"])

    if not alive:
        return (
            '<tr><td colspan="9" style="text-align:center;color:var(--text-dim);">'
            "No qualifying submissions found — requires 3/3 seeds, known artifact, BPB &ge; 0.5.</td></tr>"
        )

    return "\n".join(_build_row(rank, pr) for rank, pr in enumerate(alive, start=1))


_STATUS_SORT_ORDER = {"ALIVE": 0, "AT_RISK": 1, "INCOMPLETE": 2, "DEAD": 3, "UNKNOWN": 4}

def _build_archive_rows(prs: list[dict[str, Any]]) -> str:
    """Build tbody rows for the All Submissions archive.

    Shows ALL PRs, sorted ALIVE-first then by BPB ascending within each group.
    """
    all_prs = list(prs)
    all_prs.sort(key=lambda p: (
        _STATUS_SORT_ORDER.get(p.get("status", "UNKNOWN"), 4),
        p.get("bpb") if p.get("bpb") is not None else 999,
    ))

    if not all_prs:
        return (
            '<tr><td colspan="10" style="text-align:center;color:var(--text-dim);">'
            "No submissions found — check fetch/classify pipeline.</td></tr>"
        )

    return "\n".join(_build_row(rank, pr, include_reason=True) for rank, pr in enumerate(all_prs, start=1))


# ---------------------------------------------------------------------------
# HTML injection helpers
# ---------------------------------------------------------------------------

def _inject_sentinels(html: str) -> str:
    """Add AGORA sentinel comments to leaderboard, techniques, and timeline blocks."""
    markers = (
        NEURAL_START,
        NEURAL_END,
        ARCHIVE_START,
        ARCHIVE_END,
        TECHNIQUES_START,
        TECHNIQUES_END,
        TIMELINE_START,
        TIMELINE_END,
    )
    if all(marker in html for marker in markers):
        return html

    # State machine: find first tbody after each header
    lines = html.split("\n")
    new_lines: list[str] = []

    neural_header_seen = False
    neural_tbody_open = False
    neural_done = False

    archive_header_seen = False
    archive_tbody_open = False
    archive_done = False

    for line in lines:
        stripped = line.strip()

        # Detect section headers
        if "Neural-Only" in line and not neural_done:
            neural_header_seen = True
        if "All Submissions" in line and not archive_done:
            archive_header_seen = True

        # Neural tbody injection
        if neural_header_seen and not neural_done and not archive_header_seen:
            if stripped == "<tbody>" and not neural_tbody_open:
                new_lines.append(line)
                new_lines.append(NEURAL_START)
                neural_tbody_open = True
                continue
            if stripped == "</tbody>" and neural_tbody_open:
                new_lines.append(NEURAL_END)
                new_lines.append(line)
                neural_tbody_open = False
                neural_done = True
                continue

        # Archive tbody injection
        if archive_header_seen and not archive_done:
            if stripped == "<tbody>" and not archive_tbody_open:
                new_lines.append(line)
                new_lines.append(ARCHIVE_START)
                archive_tbody_open = True
                continue
            if stripped == "</tbody>" and archive_tbody_open:
                new_lines.append(ARCHIVE_END)
                new_lines.append(line)
                archive_tbody_open = False
                archive_done = True
                continue

        new_lines.append(line)

    result = "\n".join(new_lines)

    if TECHNIQUES_START not in result:
        result = re.sub(
            r'(<section id="techniques">\s*<h2>.*?</h2>\s*)(.*?)(\s*</section>)',
            lambda match: (
                f"{match.group(1)}{TECHNIQUES_START}\n"
                f"{(match.group(2).strip() or '<p style=\"color:var(--text-dim);\">Technique map pending.</p>')}\n"
                f"{TECHNIQUES_END}{match.group(3)}"
            ),
            result,
            count=1,
            flags=re.DOTALL,
        )

    if TIMELINE_START not in result:
        result = re.sub(
            r'(<section id="timeline">\s*<h2>.*?</h2>\s*)(.*?)(\s*</section>)',
            lambda match: (
                f"{match.group(1)}{TIMELINE_START}\n"
                f"{(match.group(2).strip() or '<p style=\"color:var(--text-dim);\">Timeline pending.</p>')}\n"
                f"{TIMELINE_END}{match.group(3)}"
            ),
            result,
            count=1,
            flags=re.DOTALL,
        )

    if NEURAL_START not in result:
        raise RuntimeError(
            "Failed to inject Neural-Only sentinel — header or tbody not found. "
            "Check that 'Neural-Only' h3 and its <tbody> exist in index.html."
        )
    if ARCHIVE_START not in result:
        raise RuntimeError(
            "Failed to inject Archive sentinel — header or tbody not found. "
            "Check that 'All Submissions' h3 and its <tbody> exist in index.html."
        )
    if TECHNIQUES_START not in result or TECHNIQUES_END not in result:
        raise RuntimeError("Failed to inject Techniques sentinels — section not found.")
    if TIMELINE_START not in result or TIMELINE_END not in result:
        raise RuntimeError("Failed to inject Timeline sentinels — section not found.")

    return result


def _replace_between_sentinels(html: str, start_marker: str, end_marker: str, new_content: str) -> str:
    """Replace content between start_marker and end_marker with new_content."""
    pattern = re.compile(
        re.escape(start_marker) + r".*?" + re.escape(end_marker),
        re.DOTALL,
    )
    replacement = f"{start_marker}\n{new_content}\n{end_marker}"
    result, count = pattern.subn(replacement, html)
    if count == 0:
        raise RuntimeError(f"Sentinel markers not found in HTML: {start_marker!r}")
    if count > 1:
        raise RuntimeError(f"Multiple occurrences of sentinel found: {start_marker!r}")
    return result


def _update_version_bar(html: str, now: datetime) -> str:
    """Legacy — replaced by _update_version_bar_from_changelog. Kept for compat."""
    return html


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point."""
    # --- Step 1: Fetch PRs ---
    fetch_ok = _run_script("fetch_prs.py")
    if not fetch_ok:
        print("[PIPELINE] fetch_prs.py failed — attempting to use cached data", flush=True)
        if not CACHE_PATH.exists():
            print("[FATAL] No cache available. Cannot continue.", flush=True)
            sys.exit(1)

    # --- Step 2: Fetch community issue-form submissions ---
    community_ok = _run_script("fetch_community.py")
    if not community_ok:
        print(
            "[PIPELINE] fetch_community.py failed — continuing with existing community data",
            flush=True,
        )

    # --- Step 3: Classify PRs ---
    classify_ok = _run_script("classify.py")
    if not classify_ok:
        print("[FATAL] classify.py failed. Cannot build leaderboard.", flush=True)
        sys.exit(1)

    techniques_ok = _run_script("techniques.py")
    if not techniques_ok:
        print("[FATAL] techniques.py failed. Cannot build technique map.", flush=True)
        sys.exit(1)

    timeline_svg = _run_script("timeline.py", capture_stdout=True)
    if not timeline_svg:
        print("[FATAL] timeline.py failed. Cannot build BPB timeline.", flush=True)
        sys.exit(1)

    # --- Step 4: Load classified cache ---
    print(f"[PIPELINE] Loading classified cache from {CACHE_PATH}", flush=True)
    with CACHE_PATH.open("r", encoding="utf-8") as f:
        cache: dict[str, Any] = json.load(f)
    prs: list[dict[str, Any]] = cache.get("prs", [])
    print(f"[PIPELINE] {len(prs)} classified PRs loaded", flush=True)

    print(f"[PIPELINE] Loading technique payload from {TECHNIQUES_PATH}", flush=True)
    with TECHNIQUES_PATH.open("r", encoding="utf-8") as f:
        techniques_payload: dict[str, Any] = json.load(f)

    # Load data-driven section JSON files (with fallback for missing files)
    def _load_json(path: Path, default: Any, label: str) -> Any:
        if path.exists():
            with path.open("r", encoding="utf-8") as fj:
                return json.load(fj)
        print(f"[WARN] {path} not found — {label} section will be empty", flush=True)
        return default

    print("[PIPELINE] Loading data-driven section files...", flush=True)
    checklist_data = _load_json(CHECKLIST_PATH, {"items": [], "techniques": {}}, "checklist")
    rulings_data = _load_json(RULINGS_PATH, [], "rulings")
    alerts_data = _load_json(ALERTS_PATH, [], "alerts")
    compute_data = _load_json(COMPUTE_PATH, {"golden_rule": {}, "providers": [], "smoke_tests": [], "hardware_comparison": {}}, "compute")

    # Stats summary
    status_counts: dict[str, int] = {}
    for pr in prs:
        s = pr.get("status", "UNKNOWN")
        status_counts[s] = status_counts.get(s, 0) + 1
    print(f"[PIPELINE] Status distribution: {status_counts}", flush=True)

    # --- Step 4: Read index.html ---
    if not INDEX_HTML_PATH.exists():
        print(f"[FATAL] {INDEX_HTML_PATH} not found", flush=True)
        sys.exit(1)

    html = INDEX_HTML_PATH.read_text(encoding="utf-8")

    # --- Step 5: Inject sentinels (idempotent) ---
    print("[PIPELINE] Injecting/verifying sentinel comments...", flush=True)
    try:
        html = _inject_sentinels(html)
    except RuntimeError as exc:
        print(f"[FATAL] Sentinel injection failed: {exc}", flush=True)
        sys.exit(1)

    # --- Step 6: Build and inject leaderboard rows ---
    print("[PIPELINE] Building Neural-Only leaderboard rows...", flush=True)
    neural_rows = _build_neural_rows(prs)

    print("[PIPELINE] Building All Submissions archive rows...", flush=True)
    archive_rows = _build_archive_rows(prs)

    print("[PIPELINE] Building technique map HTML...", flush=True)
    techniques_html = _render_techniques_section(techniques_payload)

    print("[PIPELINE] Building timeline HTML...", flush=True)
    timeline_html = _render_timeline_section(str(timeline_svg).strip(), prs)

    print("[PIPELINE] Building checklist, rulings, alerts, compute guide HTML...", flush=True)
    checklist_html = _render_checklist(checklist_data)
    rulings_html = _render_rulings(rulings_data)
    alerts_html = _render_alerts(alerts_data)
    compute_html = _render_compute_guide(compute_data)

    try:
        html = _replace_between_sentinels(html, NEURAL_START, NEURAL_END, neural_rows)
        html = _replace_between_sentinels(html, ARCHIVE_START, ARCHIVE_END, archive_rows)
        html = _replace_between_sentinels(html, TECHNIQUES_START, TECHNIQUES_END, techniques_html)
        html = _replace_between_sentinels(html, TIMELINE_START, TIMELINE_END, timeline_html)
        html = _replace_between_sentinels(html, CHECKLIST_START, CHECKLIST_END, checklist_html)
        html = _replace_between_sentinels(html, RULINGS_START, RULINGS_END, rulings_html)
        html = _replace_between_sentinels(html, ALERTS_START, ALERTS_END, alerts_html)
        html = _replace_between_sentinels(html, COMPUTE_START, COMPUTE_END, compute_html)
    except RuntimeError as exc:
        print(f"[FATAL] Table replacement failed: {exc}", flush=True)
        sys.exit(1)

    # --- Step 7: Render changelog + sync version bar ---
    now = datetime.now(timezone.utc)
    changelog: list[dict[str, Any]] = []
    if CHANGELOG_PATH.exists():
        with CHANGELOG_PATH.open("r", encoding="utf-8") as f:
            changelog = json.load(f)
        print(f"[PIPELINE] Loaded {len(changelog)} changelog entries", flush=True)
        changelog_html = _render_changelog(changelog)
        if CHANGELOG_START in html and CHANGELOG_END in html:
            html = _replace_between_sentinels(html, CHANGELOG_START, CHANGELOG_END, changelog_html)
            print("[PIPELINE] Changelog rendered from data/changelog.json", flush=True)
        html = _update_version_bar_from_changelog(html, changelog, now)
        print(f"[PIPELINE] Version bar synced to v{changelog[0]['version']}", flush=True)
    else:
        print("[WARN] data/changelog.json not found — skipping changelog render", flush=True)
        html = _update_version_bar(html, now)

    # Also update the seeded-data notice paragraphs
    html = html.replace(
        "Showing seeded data. Full API-powered leaderboard coming in Phase 2.",
        f"Live data from GitHub API. Last updated: {now.strftime('%B %-d, %Y')}.",
    )
    html = html.replace(
        "Showing seeded data. Full archive coming in Phase 2.",
        f"Live data. {len(prs)} PRs tracked. <a href=\"https://github.com/MatoTeziTanka/parameter-golf/issues/new?template=correction.yml\">Dispute a classification</a>.",
    )

    # --- Step 8: Write updated index.html ---
    print(f"[PIPELINE] Writing updated index.html ({len(html):,} bytes)", flush=True)
    # Write atomically
    tmp_path = INDEX_HTML_PATH.with_suffix(".html.tmp")
    tmp_path.write_text(html, encoding="utf-8")
    tmp_path.replace(INDEX_HTML_PATH)

    # --- Done ---
    alive_count = status_counts.get("ALIVE", 0)
    print(
        f"[DONE] Pipeline complete. "
        f"ALIVE: {alive_count} | DEAD: {status_counts.get('DEAD', 0)} | "
        f"AT_RISK: {status_counts.get('AT_RISK', 0)} | "
        f"INCOMPLETE: {status_counts.get('INCOMPLETE', 0)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
