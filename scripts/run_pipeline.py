#!/usr/bin/env python3
"""
run_pipeline.py — Orchestrator for AGORA Phase 2 leaderboard pipeline.

Steps:
  1. Run fetch_prs.py to update data/pr_cache.json
  2. Run classify.py to classify all PRs
  3. Read index.html
  4. Replace Neural-Only leaderboard tbody with ALIVE PRs sorted by BPB asc
  5. Replace All Submissions archive tbody with ALL PRs sorted by BPB asc
  6. Update the version bar timestamp
  7. Write updated index.html

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
    GITHUB_TOKEN — passed through to fetch_prs.py
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
CACHE_PATH = REPO_ROOT / "data" / "pr_cache.json"
INDEX_HTML_PATH = REPO_ROOT / "index.html"
SCRIPTS_DIR = Path(__file__).parent

NEURAL_START = "<!-- AGORA:LEADERBOARD_NEURAL_START -->"
NEURAL_END = "<!-- AGORA:LEADERBOARD_NEURAL_END -->"
ARCHIVE_START = "<!-- AGORA:LEADERBOARD_ARCHIVE_START -->"
ARCHIVE_END = "<!-- AGORA:LEADERBOARD_ARCHIVE_END -->"

PR_BASE_URL = "https://github.com/openai/parameter-golf/pull"


# ---------------------------------------------------------------------------
# Step 1 & 2: Run sub-scripts
# ---------------------------------------------------------------------------

def _run_script(script_name: str) -> bool:
    """Run a script in the scripts/ directory. Returns True if success."""
    script_path = SCRIPTS_DIR / script_name
    env = dict(os.environ)
    print(f"[PIPELINE] Running {script_name}...", flush=True)
    result = subprocess.run(
        [sys.executable, str(script_path)],
        env=env,
        capture_output=False,  # let stdout/stderr flow through
    )
    if result.returncode != 0:
        print(f"[PIPELINE] {script_name} exited with code {result.returncode}", flush=True)
        return False
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
    """Render an HTML badge for a PR status."""
    badges: dict[str, tuple[str, str]] = {
        "ALIVE": ("badge-alive", "ALIVE"),
        "DEAD": ("badge-dead", "DEAD"),
        "AT_RISK": ("badge-risk", "AT-RISK"),
        "INCOMPLETE": ("badge-incomplete", "INCOMPLETE"),
        "UNKNOWN": ("badge-incomplete", "UNKNOWN"),
    }
    if status == "INCOMPLETE" and seeds is not None and seeds < 3:
        cls, label = "badge-incomplete", f"{seeds} SEED{'S' if seeds != 1 else ''}"
    else:
        cls, label = badges.get(status, ("badge-incomplete", status))
    return f'<span class="badge {cls}">{label}</span>'


def _pr_link(number: int) -> str:
    """Render a PR number as an HTML link."""
    return f'<a href="{PR_BASE_URL}/{number}">#{number}</a>'


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

def _build_neural_rows(prs: list[dict[str, Any]]) -> str:
    """Build tbody rows for the Neural-Only leaderboard.

    Shows ALIVE PRs only, sorted by BPB ascending (lower = better).
    """
    alive = [
        p for p in prs
        if p.get("status") == "ALIVE" and p.get("bpb") is not None
    ]
    alive.sort(key=lambda p: p["bpb"])

    if not alive:
        return (
            '<tr><td colspan="7" style="text-align:center;color:var(--text-dim);">'
            "No ALIVE submissions found — check fetch/classify pipeline.</td></tr>"
        )

    rows: list[str] = []
    for rank, pr in enumerate(alive, start=1):
        num = pr["number"]
        author = pr.get("author", "unknown")
        bpb = _format_bpb(pr.get("bpb"))
        seeds = _format_seeds(pr.get("seeds"))
        artifact = _format_artifact(pr.get("artifact_bytes"))
        badge = _status_badge("ALIVE", pr.get("seeds"))
        rows.append(
            f"<tr>"
            f"<td>{rank}</td>"
            f"<td>{_pr_link(num)}</td>"
            f"<td>@{author}</td>"
            f"<td>{bpb}</td>"
            f"<td>{seeds}</td>"
            f"<td>{artifact}</td>"
            f"<td>{badge}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def _build_archive_rows(prs: list[dict[str, Any]]) -> str:
    """Build tbody rows for the All Submissions archive.

    Shows ALL PRs with a BPB, sorted by BPB ascending.
    Color-coded by status.
    """
    with_bpb = [p for p in prs if p.get("bpb") is not None]
    without_bpb = [p for p in prs if p.get("bpb") is None]

    # Sort PRs with BPB ascending, then append those without BPB
    with_bpb.sort(key=lambda p: p["bpb"])
    all_prs = with_bpb + without_bpb

    if not all_prs:
        return (
            '<tr><td colspan="7" style="text-align:center;color:var(--text-dim);">'
            "No submissions found — check fetch/classify pipeline.</td></tr>"
        )

    rows: list[str] = []
    for rank, pr in enumerate(all_prs, start=1):
        num = pr["number"]
        author = pr.get("author", "unknown")
        bpb = _format_bpb(pr.get("bpb"))
        seeds = _format_seeds(pr.get("seeds"))
        status = pr.get("status", "UNKNOWN")
        badge = _status_badge(status, pr.get("seeds"))
        reason = "&mdash;" if status == "ALIVE" else _dead_reason(pr)
        rows.append(
            f"<tr>"
            f"<td>{rank}</td>"
            f"<td>{badge}</td>"
            f"<td>{_pr_link(num)}</td>"
            f"<td>@{author}</td>"
            f"<td>{bpb}</td>"
            f"<td>{seeds}</td>"
            f"<td>{reason}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# HTML injection helpers
# ---------------------------------------------------------------------------

def _inject_sentinels(html: str) -> str:
    """Add AGORA sentinel comments around the two tbody blocks on first run.

    Approach:
      - Find the tbody immediately following the 'Neural-Only' h3
      - Find the tbody immediately following the 'All Submissions' h3
      - Wrap each tbody content in sentinel comments

    If sentinels already exist, return html unchanged.
    """
    if NEURAL_START in html and ARCHIVE_START in html:
        return html  # already injected, nothing to do

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
    """Update the 'Last updated' timestamp in the version bar."""
    date_str = now.strftime("%B %-d, %Y")  # e.g. "March 29, 2026"
    # Pattern: Last updated: <date text> (up to the next middot or angle bracket)
    updated = re.sub(
        r"(Last updated:\s*)[^&<]+",
        rf"\g<1>{date_str} ",
        html,
        count=1,
    )
    # Also bump version to v0.2.0 to reflect Phase 2 activation
    updated = re.sub(
        r"<strong style=\"color:var\(--accent\);\">v[0-9.]+</strong>",
        '<strong style="color:var(--accent);">v0.2.0</strong>',
        updated,
        count=1,
    )
    # Update "Phase 1 (static)" to "Phase 2 (live API)"
    updated = updated.replace("Phase 1 (static)", "Phase 2 (live API)", 1)
    return updated


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

    # --- Step 2: Classify PRs ---
    classify_ok = _run_script("classify.py")
    if not classify_ok:
        print("[FATAL] classify.py failed. Cannot build leaderboard.", flush=True)
        sys.exit(1)

    # --- Step 3: Load classified cache ---
    print(f"[PIPELINE] Loading classified cache from {CACHE_PATH}", flush=True)
    with CACHE_PATH.open("r", encoding="utf-8") as f:
        cache: dict[str, Any] = json.load(f)
    prs: list[dict[str, Any]] = cache.get("prs", [])
    print(f"[PIPELINE] {len(prs)} classified PRs loaded", flush=True)

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

    try:
        html = _replace_between_sentinels(html, NEURAL_START, NEURAL_END, neural_rows)
        html = _replace_between_sentinels(html, ARCHIVE_START, ARCHIVE_END, archive_rows)
    except RuntimeError as exc:
        print(f"[FATAL] Table replacement failed: {exc}", flush=True)
        sys.exit(1)

    # --- Step 7: Update version bar ---
    now = datetime.now(timezone.utc)
    print(f"[PIPELINE] Updating version bar timestamp to {now.date()}", flush=True)
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
