#!/usr/bin/env python3
"""
classify.py — Compliance classification engine for parameter-golf PRs.

Reads data/pr_cache.json, classifies each PR as DEAD / AT_RISK / INCOMPLETE / ALIVE,
and writes the results back to the same cache file (adds status, flags, confidence,
track, and technique_type fields).

Status definitions:
  DEAD       — Closed by a maintainer (valerio-oai, 0hq, cocohearts) with a
               compliance-terminating comment, or CLOSED/MERGED state with
               maintainer mention of banned technique.
  AT_RISK    — OPEN but body or file paths contain banned-technique signals.
  INCOMPLETE — OPEN but missing required fields (BPB, seeds < 3, no submission.json,
               or artifact unknown).
  ALIVE      — OPEN, has BPB, 3 seeds confirmed, artifact known, passes compliance.

Track definitions:
  record     — Main 10min/16MB competition record track (default).
  non-record — Explicitly non-record (title contains "non-record"/"non record"/
               "notable", or file path contains "track_non_record").
  unknown    — Cannot determine from available signals.

Technique type definitions:
  neural   — Pure neural architecture (default if no other signals).
  cache    — Uses n-gram or other caching (ngram/cache/backoff signals).
  ttt      — Uses test-time training (TTT/test-time signals).
  hybrid   — Mix of neural + cache and/or TTT.
  unknown  — Cannot determine from available signals.

Usage:
    python scripts/classify.py

Input:  data/pr_cache.json
Output: data/pr_cache.json  (status/flags/confidence/track/technique_type added in-place)
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_PATH = Path(__file__).parent.parent / "data" / "pr_cache.json"

MAINTAINER_LOGINS: frozenset[str] = frozenset({"valerio-oai", "0hq", "cocohearts"})

# File path substrings that indicate non-record track
_NON_RECORD_PATH_SUBSTRINGS: tuple[str, ...] = ("track_non_record",)

# Title substrings (case-insensitive) that indicate non-record track
_NON_RECORD_TITLE_SUBSTRINGS: tuple[str, ...] = (
    "non-record",
    "non record",
    "notable non-record",
    "notable",
)

# Keywords indicating cache/n-gram technique in title, body, or compliance_keywords
_CACHE_SIGNALS: tuple[str, ...] = (
    "ngram",
    "n-gram",
    "n_gram",
    "backoff",
    "cache",
)

# Keywords indicating TTT technique
_TTT_SIGNALS: tuple[str, ...] = (
    "ttt",
    "test-time",
    "test_time",
    "test time",
)

# Keywords in maintainer comments that signal definitive closure
DEAD_COMMENT_PATTERNS: list[str] = [
    "disallowed",
    "illegal",
    "closed for now",
    "not allowed",
    "hashed n-gram",
    "two-pass",
    "two_pass",
    "closing",
    "close this",
    "invalid",
    "banned",
    "does not comply",
    "ngram",
    "n-gram cache",
]

# Positive acceptance language in maintainer comments (indicates MERGED = accepted)
ACCEPTANCE_PATTERNS: list[str] = [
    "merging into the leaderboard",
    "merging this",
    "looks legal",
    "clears the",
    "approved",
    "accepted",
    "well done",
    "nice work",
    "congratulations",
]

# Keywords in PR body that indicate AT_RISK (banned technique signals)
AT_RISK_BODY_PATTERNS: list[tuple[str, str]] = [
    # (regex_pattern, human_label)
    (r"\bngram\b|\bn.gram\b|\bhashed.n.gram", "ngram-body"),
    (r"\btwo.pass\b|\b2.pass\b|\brescor", "two-pass-body"),
    (r"\bNGRAM_TWO_PASS\b", "NGRAM_TWO_PASS-body"),
    (r"\bprefill\b.*\btrain\b|\btrain\b.*\bprefill\b", "prefill+train-body"),
    (r"\bgptq\b.*\bcalibrat\b.*\beval\b|\bgptq\b.*\beval\b.*\bcalibrat\b", "gptq-calibrate-eval-body"),
]

# File path patterns that indicate AT_RISK
AT_RISK_PATH_PATTERNS: list[tuple[str, str]] = [
    (r"ngram|n_gram", "ngram-path"),
    (r"backoff", "backoff-path"),
    # "cache" is only flagged when it appears to be eval-time caching
    # We look for it adjacent to eval/test in the path
    (r"(?:eval|test|val).*cache|cache.*(?:eval|test|val)", "eval-cache-path"),
]


# ---------------------------------------------------------------------------
# Track and technique classification helpers
# ---------------------------------------------------------------------------

def _classify_track(pr: dict[str, Any]) -> str:
    """Classify the competition track for a PR.

    Returns one of: "record", "non-record", "unknown".

    Detection priority:
      1. File paths containing "track_non_record" -> non-record
      2. Title (case-insensitive) containing non-record signals -> non-record
      3. Default -> record
    """
    title: str = (pr.get("title") or "").lower()
    file_paths: list[str] = pr.get("file_paths", [])
    path_str = " ".join(file_paths).lower()

    # Check file paths first (highest signal)
    for substring in _NON_RECORD_PATH_SUBSTRINGS:
        if substring in path_str:
            return "non-record"

    # Check title
    for substring in _NON_RECORD_TITLE_SUBSTRINGS:
        if substring in title:
            return "non-record"

    return "record"


def _classify_technique_type(pr: dict[str, Any]) -> str:
    """Classify the technique type for a PR.

    Returns one of: "neural", "cache", "ttt", "hybrid", "unknown".

    Detection:
      - Checks title, compliance_keywords list, and body text for cache and TTT signals.
      - If both cache and TTT signals found -> hybrid.
      - If only cache signals -> cache.
      - If only TTT signals -> ttt.
      - Default -> neural.
    """
    title: str = (pr.get("title") or "").lower()
    body: str = (pr.get("body") or "").lower()
    compliance_keywords: list[str] = [k.lower() for k in pr.get("compliance_keywords", [])]
    file_paths: list[str] = pr.get("file_paths", [])
    path_str = " ".join(file_paths).lower()

    # Build a combined signal corpus from all text sources
    combined = f"{title} {body} {' '.join(compliance_keywords)} {path_str}"

    has_cache = any(sig in combined for sig in _CACHE_SIGNALS)
    has_ttt = any(sig in combined for sig in _TTT_SIGNALS)

    if has_cache and has_ttt:
        return "hybrid"
    if has_cache:
        return "cache"
    if has_ttt:
        return "ttt"
    return "neural"


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

def _is_dead(pr: dict[str, Any]) -> tuple[bool, list[str], str]:
    """Determine if a PR should be classified DEAD.

    Returns (is_dead, flags, confidence).
    Confidence: HIGH if maintainer explicitly named a banned technique.
    """
    flags: list[str] = []
    state = pr.get("state", "OPEN")
    maintainer_comments: list[str] = pr.get("maintainer_comments", [])

    # MERGED PRs: accepted to official leaderboard if maintainer used acceptance language
    if state == "MERGED":
        for comment in maintainer_comments:
            comment_lower = comment.lower()
            # First check for compliance-terminating language (takes precedence)
            for pat in DEAD_COMMENT_PATTERNS:
                if pat.lower() in comment_lower:
                    flags.append(f"maintainer-closed:{pat}")
                    return True, flags, "HIGH"
            # Then check for positive acceptance language
            for pat in ACCEPTANCE_PATTERNS:
                if pat.lower() in comment_lower:
                    # Accepted to leaderboard — not dead
                    return False, [], "NONE"
        # MERGED with maintainer comment but no clear signal — treat as ALIVE (benefit of doubt)
        # because MERGED typically means maintainer accepted it
        return False, [], "NONE"

    # CLOSED PRs: compliance termination
    if state == "CLOSED" and maintainer_comments:
        # Check if maintainer comment contains a compliance-terminating keyword
        for comment in maintainer_comments:
            comment_lower = comment.lower()
            for pat in DEAD_COMMENT_PATTERNS:
                if pat.lower() in comment_lower:
                    flags.append(f"maintainer-closed:{pat}")
                    return True, flags, "HIGH"

        # CLOSED + any maintainer comment = likely dead (MEDIUM confidence)
        flags.append("closed-with-maintainer-comment")
        return True, flags, "MEDIUM"

    # OPEN PR with maintainer comment containing closure keywords = DEAD
    if state == "OPEN":
        for comment in maintainer_comments:
            comment_lower = comment.lower()
            for pat in DEAD_COMMENT_PATTERNS:
                if pat.lower() in comment_lower:
                    flags.append(f"maintainer-closure-keyword:{pat}")
                    return True, flags, "HIGH"

    return False, flags, "NONE"


def _is_at_risk(pr: dict[str, Any]) -> tuple[bool, list[str], str]:
    """Determine if an OPEN PR has compliance risk signals.

    Returns (is_at_risk, flags, confidence).
    """
    flags: list[str] = []
    body: str = (pr.get("title") or "") + "\n" + (pr.get("body") or "")
    # Note: body field from cache is actually the raw PR body.
    # The pr record from parse_pr_node doesn't preserve raw body — we use compliance_keywords
    # and technique_summary as proxies. However classify.py should also check the
    # compliance_keywords extracted during fetch.

    compliance_keywords: list[str] = pr.get("compliance_keywords", [])
    file_paths: list[str] = pr.get("file_paths", [])

    # Check compliance keywords extracted during fetch (HIGH confidence signals)
    high_risk_keywords = {"ngram", "ngram_path", "two_pass"}
    for kw in compliance_keywords:
        if kw in high_risk_keywords:
            flags.append(f"keyword:{kw}")

    # Check file paths for AT_RISK patterns
    path_str = " ".join(file_paths).lower()
    for pattern, label in AT_RISK_PATH_PATTERNS:
        if re.search(pattern, path_str, re.IGNORECASE):
            flags.append(f"path:{label}")

    # Confidence based on where signal was found
    if not flags:
        return False, flags, "NONE"

    path_flags = [f for f in flags if f.startswith("path:")]
    if path_flags:
        return True, flags, "MEDIUM"
    return True, flags, "LOW"


def _is_incomplete(pr: dict[str, Any]) -> tuple[bool, list[str]]:
    """Determine if a PR is INCOMPLETE (missing required submission fields).

    A PR must have ALL of the following to be ALIVE:
      - bpb field present (score extracted)
      - seeds == 3 (all three seeds run: 42, 1337, 2024)
      - artifact_size field present and not None/unknown (artifact confirmed)
      - submission.json present in file paths

    Returns (is_incomplete, flags).
    """
    flags: list[str] = []

    bpb = pr.get("bpb")
    if bpb is None:
        flags.append("no-bpb")

    seeds = pr.get("seeds")
    if seeds is None:
        flags.append("seeds-unknown")
    elif seeds < 3:
        flags.append(f"seeds-only-{seeds}")

    # Artifact must be known — None means it was never parsed from the PR body.
    # The pipeline sets artifact_size=None when the artifact field was "?" or missing.
    artifact_size = pr.get("artifact_size")
    if artifact_size is None:
        flags.append("artifact-unknown")

    file_paths = pr.get("file_paths", [])
    has_submission_json = any(
        "submission.json" in p.lower() for p in file_paths
    )
    if not has_submission_json:
        flags.append("no-submission-json")

    return bool(flags), flags


def classify_pr(pr: dict[str, Any]) -> dict[str, Any]:
    """Classify a single PR and return a copy with status/flags/confidence/track/technique_type added."""
    result = dict(pr)
    state = pr.get("state", "OPEN")

    # Classify track and technique type for all PRs (independent of status)
    result["track"] = _classify_track(pr)
    result["technique_type"] = _classify_technique_type(pr)

    # Gate 1: DEAD check (highest priority)
    dead, dead_flags, dead_confidence = _is_dead(pr)
    if dead:
        result["status"] = "DEAD"
        result["flags"] = dead_flags
        result["confidence"] = dead_confidence
        return result

    # MERGED PRs that passed the DEAD check were accepted to the official leaderboard
    if state == "MERGED":
        result["status"] = "ALIVE"
        result["flags"] = ["merged-to-leaderboard"]
        result["confidence"] = "HIGH"
        return result

    # CLOSED PRs without a maintainer comment — treat as abandoned/dead
    if state == "CLOSED":
        result["status"] = "DEAD"
        result["flags"] = ["closed-no-maintainer-comment"]
        result["confidence"] = "LOW"
        return result

    # Gate 2: AT_RISK check (OPEN PRs with compliance signals)
    at_risk, risk_flags, risk_confidence = _is_at_risk(pr)
    if at_risk:
        result["status"] = "AT_RISK"
        result["flags"] = risk_flags
        result["confidence"] = risk_confidence
        return result

    # Gate 3: INCOMPLETE check — seeds must be 3, bpb present, artifact known
    incomplete, incomplete_flags = _is_incomplete(pr)
    if incomplete:
        result["status"] = "INCOMPLETE"
        result["flags"] = incomplete_flags
        result["confidence"] = "HIGH"  # missing data is deterministic
        return result

    # Gate 4: BPB floor check — ALIVE PRs with impossibly low BPB for neural-only
    # Neural-only approaches cannot achieve BPB < 0.5; flag as AT_RISK for manual review.
    bpb = pr.get("bpb")
    if bpb is not None and bpb > 0 and bpb < 0.5:
        result["status"] = "AT_RISK"
        result["flags"] = ["suspiciously-low-bpb"]
        result["confidence"] = "MEDIUM"
        result["flag_reason"] = "Suspiciously low BPB — likely non-neural"
        return result

    # Gate 5: ALIVE — passed all checks
    result["status"] = "ALIVE"
    result["flags"] = []
    result["confidence"] = "HIGH"
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point."""
    if not CACHE_PATH.exists():
        print(f"[FATAL] Cache not found at {CACHE_PATH}. Run fetch_prs.py first.", flush=True)
        sys.exit(1)

    print(f"[INFO] Loading cache from {CACHE_PATH}", flush=True)
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            cache: dict[str, Any] = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[FATAL] Failed to read cache: {exc}", flush=True)
        sys.exit(1)

    prs: list[dict[str, Any]] = cache.get("prs", [])
    print(f"[INFO] Classifying {len(prs)} PRs...", flush=True)

    classified: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {"ALIVE": 0, "AT_RISK": 0, "INCOMPLETE": 0, "DEAD": 0}
    track_counts: dict[str, int] = {}
    technique_counts: dict[str, int] = {}

    for pr in prs:
        try:
            result = classify_pr(pr)
            classified.append(result)
            status = result.get("status", "UNKNOWN")
            status_counts[status] = status_counts.get(status, 0) + 1
            track = result.get("track", "unknown")
            track_counts[track] = track_counts.get(track, 0) + 1
            technique = result.get("technique_type", "unknown")
            technique_counts[technique] = technique_counts.get(technique, 0) + 1
        except Exception as exc:
            print(f"[WARN] Failed to classify PR #{pr.get('number', '?')}: {exc}", flush=True)
            pr["status"] = "UNKNOWN"
            pr["flags"] = [f"classification-error:{exc}"]
            pr["confidence"] = "NONE"
            pr["track"] = "unknown"
            pr["technique_type"] = "unknown"
            classified.append(pr)

    cache["prs"] = classified

    # Write back
    tmp_path = CACHE_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    tmp_path.replace(CACHE_PATH)

    print("[DONE] Classification complete:", flush=True)
    print("  Status breakdown:", flush=True)
    for status, count in sorted(status_counts.items()):
        print(f"    {status}: {count}", flush=True)
    print("  Track breakdown:", flush=True)
    for track, count in sorted(track_counts.items()):
        print(f"    {track}: {count}", flush=True)
    print("  Technique breakdown:", flush=True)
    for technique, count in sorted(technique_counts.items()):
        print(f"    {technique}: {count}", flush=True)


if __name__ == "__main__":
    main()
