#!/usr/bin/env python3
"""
_transform_html.py — One-shot HTML transformation for AGORA classification fixes.

Performs:
1. Remove INCOMPLETE rows from neural leaderboard (seeds != 3/3 OR artifact == ?)
2. Add Type column header + Neural badge to neural leaderboard rows
3. Renumber remaining neural rows from 1
4. Add Type column header + empty cell to archive leaderboard
5. Add new badge CSS classes (record, nonrecord, neural, cache, ttt, hybrid)

Preserves all AGORA marker comments exactly.

Usage:
    python scripts/_transform_html.py

Input/Output: index.html (in-place via tmp file)
"""

import re
import sys
from pathlib import Path

HTML_PATH = Path(__file__).parent.parent / "index.html"

# ---------------------------------------------------------------------------
# New CSS badge classes to inject after existing badge-incomplete
# ---------------------------------------------------------------------------

NEW_BADGE_CSS = """.badge-record { background: rgba(88,166,255,0.15); color: var(--blue); }
.badge-nonrecord { background: rgba(139,148,158,0.15); color: var(--text-dim); }
.badge-neural { background: rgba(188,140,255,0.15); color: var(--accent); }
.badge-cache { background: rgba(248,81,73,0.15); color: var(--red); }
.badge-ttt { background: rgba(210,153,34,0.15); color: var(--yellow); }
.badge-hybrid { background: rgba(63,185,80,0.15); color: var(--green); }"""

BADGE_INJECT_AFTER = '.badge-incomplete { background: rgba(139,148,158,0.15); color: var(--text-dim); }'

# ---------------------------------------------------------------------------
# Helper: determine if a neural row is incomplete
# ---------------------------------------------------------------------------

def _row_is_incomplete(row_html: str) -> tuple[bool, str]:
    """Return (is_incomplete, reason) for a neural leaderboard row."""
    cells = re.findall(r"<td>(.*?)</td>", row_html, re.DOTALL)
    if len(cells) < 6:
        return False, ""
    seeds = cells[4].strip()
    artifact = cells[5].strip()
    reasons = []
    if seeds != "3/3":
        reasons.append(f"seeds={seeds!r}")
    if artifact == "?":
        reasons.append("artifact=?")
    return bool(reasons), ", ".join(reasons)


# ---------------------------------------------------------------------------
# Transform: neural leaderboard
# ---------------------------------------------------------------------------

def _transform_neural_section(section: str) -> tuple[str, int, int]:
    """
    Remove incomplete rows, add Type column, renumber.
    Returns (new_section, kept_count, removed_count).
    """
    rows = re.findall(r"<tr>.*?</tr>", section, re.DOTALL)
    kept = []
    removed = 0

    for row in rows:
        incomplete, reason = _row_is_incomplete(row)
        if incomplete:
            removed += 1
            continue

        # Add Type cell before the Status cell (last <td>)
        # The status cell is: <td><span class="badge badge-alive">ALIVE</span></td>
        # We insert Type cell before it.
        row = re.sub(
            r'(<td><span class="badge badge-alive">ALIVE</span></td></tr>)',
            '<td><span class="badge badge-neural">Neural</span></td>'
            r'\1',
            row,
        )
        kept.append(row)

    # Renumber rows (first <td>N</td> in each row)
    renumbered = []
    for i, row in enumerate(kept, start=1):
        row = re.sub(r"^<tr><td>\d+</td>", f"<tr><td>{i}</td>", row)
        renumbered.append(row)

    return "\n".join(renumbered), len(renumbered), removed


# ---------------------------------------------------------------------------
# Transform: archive leaderboard header only
# ---------------------------------------------------------------------------

def _transform_archive_header(section: str) -> str:
    """
    Add Type column to archive header and empty Type cell to each row.
    The archive header is:
      <thead><tr><th>#</th><th>Status</th><th>PR</th><th>Author</th><th>BPP</th><th>Seeds</th><th>Reason</th></tr></thead>
    We insert <th>Type</th> before <th>Reason</th>.
    Each data row gets <td>—</td> before the last <td> (Reason).
    """
    # Update header: insert Type before Reason
    section = section.replace(
        "<th>Reason</th>",
        "<th>Type</th><th>Reason</th>",
    )

    # Update data rows: insert <td>&mdash;</td> before the last </td></tr>
    # Each row ends in: <td>REASON_CONTENT</td></tr>
    # We need to add a Type cell — but we need to insert BEFORE the last <td>
    # Strategy: find each <tr> and inject before the last <td>...</td></tr>
    def _add_type_cell(m: re.Match) -> str:
        row = m.group(0)
        # Find the last <td>...</td></tr> and insert type before it
        # Only add if this row doesn't already have a Type cell (idempotency)
        if '&mdash;</td></tr>' in row or '>—</td></tr>' in row:
            # Already has a dash cell (this is the reason cell) — skip adding duplicate
            # Count <td> tags to see if type cell already added
            td_count = row.count('<td>')
            # Archive rows have 7 columns: #, Status, PR, Author, BPB, Seeds, Reason
            # After adding Type: 8 columns
            if td_count >= 8:
                return row  # already transformed
        return re.sub(
            r'(<td>(?:&mdash;|—|.*?)</td></tr>)$',
            r'<td>—</td>\1',
            row,
        )

    rows = re.findall(r"<tr>.*?</tr>", section, re.DOTALL)
    # Rebuild section with transformed rows
    # We need to be careful not to double-transform; use a fresh approach:
    transformed_rows = []
    for row in rows:
        td_count = row.count('<td>')
        if td_count == 7:  # original 7 columns, add 8th (Type)
            # Insert <td>—</td> before the last </td></tr>
            row = re.sub(
                r"(<td>.*?</td>)(</tr>)$",
                lambda m: m.group(1) + "<td>—</td>" + m.group(2),
                row,
                count=1,
            )
        transformed_rows.append(row)

    # Replace the rows in the section
    # The section may have whitespace between rows; we need to reconstruct carefully.
    # Simplest: replace all rows in section with our transformed versions
    row_pattern = r"<tr>.*?</tr>"
    original_rows = re.findall(row_pattern, section, re.DOTALL)
    result = section
    for orig, new in zip(original_rows, transformed_rows):
        if orig != new:
            result = result.replace(orig, new, 1)
    return result


# ---------------------------------------------------------------------------
# Main transformation
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point."""
    if not HTML_PATH.exists():
        print(f"[FATAL] index.html not found at {HTML_PATH}", flush=True)
        sys.exit(1)

    print(f"[INFO] Reading {HTML_PATH}", flush=True)
    with HTML_PATH.open("r", encoding="utf-8") as f:
        content = f.read()

    # --- Step 1: Inject new CSS badge classes ---
    if "badge-record" not in content:
        if BADGE_INJECT_AFTER not in content:
            print("[WARN] badge-incomplete CSS anchor not found; CSS injection skipped", flush=True)
        else:
            content = content.replace(
                BADGE_INJECT_AFTER,
                BADGE_INJECT_AFTER + "\n" + NEW_BADGE_CSS,
            )
            print("[INFO] Injected new badge CSS classes", flush=True)
    else:
        print("[INFO] Badge CSS already present; skipping injection", flush=True)

    # --- Step 2: Update neural leaderboard header (add Type column) ---
    old_neural_header = "<thead><tr><th>#</th><th>PR</th><th>Author</th><th>BPB</th><th>Seeds</th><th>Artifact</th><th>Status</th></tr></thead>"
    new_neural_header = "<thead><tr><th>#</th><th>PR</th><th>Author</th><th>BPB</th><th>Seeds</th><th>Artifact</th><th>Type</th><th>Status</th></tr></thead>"
    if old_neural_header in content:
        content = content.replace(old_neural_header, new_neural_header, 1)
        print("[INFO] Updated neural leaderboard table header", flush=True)
    elif new_neural_header in content:
        print("[INFO] Neural header already has Type column; skipping", flush=True)
    else:
        print("[WARN] Neural leaderboard header not found in expected form", flush=True)

    # --- Step 3: Transform neural leaderboard rows ---
    neural_start_marker = "<!-- AGORA:LEADERBOARD_NEURAL_START -->"
    neural_end_marker = "<!-- AGORA:LEADERBOARD_NEURAL_END -->"

    ns = content.index(neural_start_marker) + len(neural_start_marker)
    ne = content.index(neural_end_marker)
    neural_rows_section = content[ns:ne]

    new_neural_rows, kept, removed = _transform_neural_section(neural_rows_section)

    # Wrap with newlines to match original formatting
    new_neural_rows_wrapped = "\n" + new_neural_rows + "\n"
    content = content[:ns] + new_neural_rows_wrapped + content[ne:]
    print(f"[INFO] Neural leaderboard: kept={kept}, removed={removed}", flush=True)

    # --- Step 4: Transform archive leaderboard ---
    archive_start_marker = "<!-- AGORA:LEADERBOARD_ARCHIVE_START -->"
    archive_end_marker = "<!-- AGORA:LEADERBOARD_ARCHIVE_END -->"

    # Re-find positions after step 3 modified content
    as_ = content.index(archive_start_marker) + len(archive_start_marker)
    ae = content.index(archive_end_marker)
    archive_section = content[as_:ae]

    new_archive_section = _transform_archive_header(archive_section)
    content = content[:as_] + new_archive_section + content[ae:]
    print("[INFO] Archive leaderboard: added Type column header and empty cells", flush=True)

    # --- Step 5: Write output ---
    tmp_path = HTML_PATH.with_suffix(".html.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(content)
    tmp_path.replace(HTML_PATH)
    print(f"[DONE] Written to {HTML_PATH}", flush=True)

    # --- Verification summary ---
    with HTML_PATH.open("r", encoding="utf-8") as f:
        final = f.read()

    # Verify markers preserved
    for marker in [neural_start_marker, neural_end_marker, archive_start_marker, archive_end_marker]:
        assert marker in final, f"MARKER MISSING: {marker}"
    print("[VERIFY] All AGORA markers preserved", flush=True)

    # Count neural rows remaining
    ns2 = final.index(neural_start_marker) + len(neural_start_marker)
    ne2 = final.index(neural_end_marker)
    final_neural_rows = re.findall(r"<tr>.*?</tr>", final[ns2:ne2], re.DOTALL)
    print(f"[VERIFY] Neural leaderboard rows after transform: {len(final_neural_rows)}", flush=True)

    # Confirm no 2/3 or 1/3 seeds in neural section
    neural_section_final = final[ns2:ne2]
    for bad_seeds in ["2/3", "1/3"]:
        if bad_seeds in neural_section_final:
            print(f"[WARN] Still found {bad_seeds!r} seeds in neural section!", flush=True)
        else:
            print(f"[VERIFY] No {bad_seeds!r} seeds in neural section", flush=True)

    # Confirm new badge types exist
    for badge in ["badge-record", "badge-neural", "badge-cache", "badge-ttt", "badge-hybrid", "badge-nonrecord"]:
        if badge in final:
            print(f"[VERIFY] CSS class {badge!r} present", flush=True)
        else:
            print(f"[WARN] CSS class {badge!r} NOT found!", flush=True)

    # Confirm Type column in archive header
    if "<th>Type</th><th>Reason</th>" in final:
        print("[VERIFY] Archive header has Type column", flush=True)
    else:
        print("[WARN] Archive header missing Type column", flush=True)

    # Confirm Type column in neural header
    if new_neural_header in final:
        print("[VERIFY] Neural header has Type column", flush=True)
    else:
        print("[WARN] Neural header missing Type column", flush=True)


if __name__ == "__main__":
    main()
