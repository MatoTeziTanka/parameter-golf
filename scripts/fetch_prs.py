#!/usr/bin/env python3
"""
fetch_prs.py — GitHub GraphQL scraper for openai/parameter-golf PRs.

Fetches all PRs (OPEN + CLOSED + MERGED), parses submission metadata from
PR bodies, caches results in data/pr_cache.json. On subsequent runs, only
fetches PRs updated since the last cache timestamp.

Usage:
    GITHUB_TOKEN=<token> python scripts/fetch_prs.py
    python scripts/fetch_prs.py  # works without token but rate-limited

Output: data/pr_cache.json
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_OWNER = "openai"
REPO_NAME = "parameter-golf"
GRAPHQL_URL = "https://api.github.com/graphql"
CACHE_PATH = Path(__file__).parent.parent / "data" / "pr_cache.json"
PAGE_SIZE = 100  # max allowed by GitHub GraphQL

# Maintainer logins whose comments carry compliance authority
MAINTAINER_LOGINS: frozenset[str] = frozenset({"valerio-oai", "0hq", "cocohearts"})

# GraphQL query — fetches one page of PRs ordered by updatedAt DESC
# $cursor is "" on first call, endCursor on subsequent calls
GRAPHQL_QUERY = """
query($cursor: String) {
  repository(owner: "openai", name: "parameter-golf") {
    pullRequests(
      states: [OPEN, CLOSED, MERGED],
      first: 100,
      after: $cursor,
      orderBy: {field: UPDATED_AT, direction: DESC}
    ) {
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        number
        title
        body
        createdAt
        updatedAt
        state
        author {
          login
        }
        files(first: 50) {
          nodes {
            path
          }
        }
        comments(first: 20) {
          nodes {
            author {
              login
            }
            body
            createdAt
          }
        }
      }
    }
  }
}
"""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_bpb(title: str, body: str) -> float | None:
    """Extract best BPB score from PR title or body.

    Patterns tried in order (most specific first):
      1. val_bpb: X.XXXX
      2. val_bpb=X.XXXX
      3. **X.XXXX BPB** (bold markdown)
      4. X.XXXX BPB (case-insensitive)
      5. Table row: | X.XXXX | (adjacent to BPB column headers)
    """
    combined = (title or "") + "\n" + (body or "")

    # Pattern set: key=value or key: value
    for pat in [
        r"val_bpb\s*[:=]\s*([0-9]+\.[0-9]+)",
    ]:
        m = re.search(pat, combined, re.IGNORECASE)
        if m:
            return float(m.group(1))

    # Bold markdown: **X.XXXX BPB**
    m = re.search(r"\*\*([0-9]+\.[0-9]+)\s*BPB\*\*", combined, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # Plain: X.XXXX BPB or X.XXXX bpb
    m = re.search(r"\b([0-9]+\.[0-9]{3,5})\s*BPB\b", combined, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # Table row: pipe-delimited value that looks like a BPB (0.x or 1.x range)
    # Match rows like: | 1.0945 | or | **1.0945** |
    for m in re.finditer(r"\|\s*\*{0,2}([0-9]\.[0-9]{3,5})\*{0,2}\s*\|", combined):
        val = float(m.group(1))
        if 0.0 < val < 2.5:
            return val

    return None


def _parse_seeds(body: str) -> int | None:
    """Extract seed count from PR body.

    Patterns:
      3-seed, seeds: 3, count of seed entries {42, 1337, 2024}, seeds: {42, 1337}
    """
    if not body:
        return None

    # "3-seed" pattern
    m = re.search(r"(\d+)-seed", body, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # "seeds: 3" or "num_seeds: 3"
    m = re.search(r"(?:num_)?seeds?\s*[:=]\s*(\d+)", body, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Count explicit seed values mentioned (42, 1337, 2024 are the competition seeds)
    competition_seeds = {42, 1337, 2024}
    found_seeds: set[int] = set()
    for m in re.finditer(r"\b(42|1337|2024)\b", body):
        found_seeds.add(int(m.group(1)))
    if len(found_seeds) >= 2:
        return len(found_seeds & competition_seeds)

    # "seeds: {" — count entries in braces
    m = re.search(r"seeds?\s*[:=]\s*\{([^}]+)\}", body, re.IGNORECASE)
    if m:
        entries = [e.strip() for e in m.group(1).split(",") if e.strip()]
        return len(entries)

    return None


def _parse_artifact_bytes(body: str) -> int | None:
    """Extract artifact size in bytes from PR body.

    Patterns:
      bytes_total: 15200000
      15,900,000 bytes
      15.2 MB
      artifact: 15.2MB
    """
    if not body:
        return None

    # bytes_total: N
    m = re.search(r"bytes_total\s*[:=]\s*([0-9,]+)", body, re.IGNORECASE)
    if m:
        return int(m.group(1).replace(",", ""))

    # N,NNN,NNN bytes  or  N bytes
    m = re.search(r"([0-9]{1,3}(?:,[0-9]{3})+)\s*bytes", body, re.IGNORECASE)
    if m:
        return int(m.group(1).replace(",", ""))

    # N bytes (no commas, large number)
    m = re.search(r"\b([0-9]{6,9})\s*bytes\b", body, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # XX.X MB
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*MB\b", body, re.IGNORECASE)
    if m:
        return int(float(m.group(1)) * 1_000_000)

    # artifact: size in submission.json style  "artifact_bytes": 15200000
    m = re.search(r'"artifact(?:_bytes)?"\s*:\s*([0-9]+)', body, re.IGNORECASE)
    if m:
        return int(m.group(1))

    return None


def _extract_technique_summary(title: str, body: str) -> str:
    """Build a short technique summary from title + body keywords."""
    combined = (title or "") + " " + (body or "")
    techniques: list[str] = []

    patterns = [
        (r"\bLeakyReLU.{0,4}squared\b|\bLeakyReLU.{0,2}\^2\b|\bLeakyReLU2\b", "LeakyReLU²"),
        (r"\bTTT\b|\btest.time training\b|\btest.time.train", "TTT"),
        (r"\bMuon\b", "Muon"),
        (r"\bMoE\b|\bMixture.of.Experts\b", "MoE"),
        (r"\bSSM\b|\bMamba\b|\bstate.space\b", "SSM"),
        (r"\bRWKV\b", "RWKV"),
        (r"\bGEGLU\b|\bGeGLU\b", "GeGLU"),
        (r"\bSwiGLU\b", "SwiGLU"),
        (r"\bLoRA\b", "LoRA"),
        (r"\bGPTQ\b", "GPTQ"),
        (r"\bINT6\b|\bint6\b", "INT6"),
        (r"\bINT8\b|\bint8\b", "INT8"),
        (r"\bEMA\b|\bexponential moving average\b", "EMA"),
        (r"\bn.gram|\bngram\b", "N-gram"),
        (r"\btwo.pass|\b2.pass\b", "Two-pass"),
    ]
    for pattern, label in patterns:
        if re.search(pattern, combined, re.IGNORECASE):
            techniques.append(label)

    return " + ".join(techniques) if techniques else ""


def _extract_compliance_keywords(body: str, file_paths: list[str]) -> list[str]:
    """Return list of compliance-relevant keywords found in body or file paths."""
    keywords: list[str] = []
    combined_body = (body or "").lower()
    combined_paths = " ".join(file_paths).lower()

    checks = {
        "ttt": r"\bttt\b|\btest.time.train",
        "ngram": r"\bn.gram|\bngram\b|\bhashed.n",
        "two_pass": r"\btwo.pass|\b2.pass\b|\brescor",
        "gptq_eval": r"\bgptq\b",
        "prefill": r"\bprefill\b|\bpre.fill\b",
        "cache_eval": r"\bcache\b",
        "leaky_relu": r"\bleakyrelu\b|\bleaky.relu\b",
    }
    for key, pat in checks.items():
        if re.search(pat, combined_body, re.IGNORECASE):
            keywords.append(key)

    for key, pat in {"ngram_path": r"ngram|n_gram", "cache_path": r"cache"}.items():
        if re.search(pat, combined_paths, re.IGNORECASE):
            keywords.append(key)

    return list(dict.fromkeys(keywords))  # preserve order, deduplicate


def _parse_pr_node(node: dict[str, Any]) -> dict[str, Any]:
    """Transform a raw GraphQL PR node into a normalized PR record."""
    number: int = node["number"]
    title: str = node.get("title") or ""
    body: str = node.get("body") or ""
    state: str = node.get("state", "UNKNOWN")
    created: str = node.get("createdAt", "")
    updated: str = node.get("updatedAt", "")
    author: str = (node.get("author") or {}).get("login", "unknown")

    file_paths: list[str] = [
        f["path"] for f in (node.get("files") or {}).get("nodes", [])
        if f and f.get("path")
    ]

    raw_comments = (node.get("comments") or {}).get("nodes", [])
    maintainer_comments: list[str] = []
    for c in raw_comments:
        if not c:
            continue
        commenter = (c.get("author") or {}).get("login", "")
        if commenter in MAINTAINER_LOGINS:
            body_text = (c.get("body") or "")[:500]  # cap at 500 chars
            maintainer_comments.append(f"{commenter}: {body_text}")

    bpb = _parse_bpb(title, body)
    seeds = _parse_seeds(body)
    artifact_bytes = _parse_artifact_bytes(body)
    technique_summary = _extract_technique_summary(title, body)
    compliance_keywords = _extract_compliance_keywords(body, file_paths)

    return {
        "number": number,
        "title": title,
        "author": author,
        "state": state,
        "created": created,
        "updated": updated,
        "bpb": bpb,
        "seeds": seeds,
        "artifact_bytes": artifact_bytes,
        "technique_summary": technique_summary,
        "file_paths": file_paths,
        "maintainer_comments": maintainer_comments,
        "compliance_keywords": compliance_keywords,
    }


# ---------------------------------------------------------------------------
# GitHub API client
# ---------------------------------------------------------------------------

def _make_session(token: str | None) -> requests.Session:
    """Build a requests Session with appropriate auth headers."""
    session = requests.Session()
    session.headers["Accept"] = "application/vnd.github+json"
    session.headers["Content-Type"] = "application/json"
    session.headers["X-GitHub-Api-Version"] = "2022-11-28"
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    else:
        print("[WARN] GITHUB_TOKEN not set — unauthenticated requests (60/hour limit)", flush=True)
    return session


def _graphql(session: requests.Session, cursor: str | None) -> dict[str, Any]:
    """Execute one paginated GraphQL call. Raises on non-200."""
    variables: dict[str, Any] = {"cursor": cursor}
    payload = {"query": GRAPHQL_QUERY, "variables": variables}
    resp = session.post(GRAPHQL_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data


def _load_cache() -> dict[str, Any]:
    """Load existing cache or return empty structure."""
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARN] Cache unreadable ({exc}), starting fresh", flush=True)
    return {"last_fetch": None, "api_calls_used": 0, "prs": []}


def _save_cache(cache: dict[str, Any]) -> None:
    """Atomically write cache to disk."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = CACHE_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    tmp_path.replace(CACHE_PATH)


# ---------------------------------------------------------------------------
# Main fetch logic
# ---------------------------------------------------------------------------

def _should_stop_early(node_updated: str, cutoff: str | None) -> bool:
    """Return True if this node was last updated before our cache cutoff.

    We order by UPDATED_AT DESC, so once we see a node older than the cutoff
    we can stop fetching more pages.
    """
    if not cutoff or not node_updated:
        return False
    return node_updated < cutoff


def fetch_all_prs(token: str | None, cache: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Fetch PRs from GitHub, returning (list_of_parsed_prs, api_calls_used)."""
    session = _make_session(token)
    cutoff: str | None = cache.get("last_fetch")

    # Build lookup of existing PRs by number for merge strategy
    existing: dict[int, dict[str, Any]] = {pr["number"]: pr for pr in cache.get("prs", [])}

    fetched: list[dict[str, Any]] = []
    cursor: str | None = None
    api_calls = 0
    page = 0
    stop_early = False

    while True:
        page += 1
        cursor_preview = repr(cursor[:20]) if cursor else "None"
        print(f"[API] Page {page} (cursor={cursor_preview})", flush=True)
        try:
            data = _graphql(session, cursor)
        except requests.HTTPError as exc:
            print(f"[ERROR] HTTP {exc.response.status_code}: {exc}", flush=True)
            if existing:
                print("[INFO] Falling back to cached data", flush=True)
                return list(existing.values()), api_calls
            raise
        except Exception as exc:
            print(f"[ERROR] API call failed: {exc}", flush=True)
            if existing:
                print("[INFO] Falling back to cached data", flush=True)
                return list(existing.values()), api_calls
            raise

        api_calls += 1
        pr_data = data["data"]["repository"]["pullRequests"]
        page_info = pr_data["pageInfo"]
        nodes = pr_data["nodes"]

        for node in nodes:
            if not node:
                continue
            updated = node.get("updatedAt", "")
            if _should_stop_early(updated, cutoff):
                stop_early = True
                break
            pr = _parse_pr_node(node)
            fetched.append(pr)

        print(f"[API] Fetched {len(fetched)} PRs so far ({api_calls} calls)", flush=True)

        if stop_early:
            print(f"[API] Reached cached cutoff ({cutoff}), stopping early", flush=True)
            break

        if not page_info["hasNextPage"]:
            print("[API] No more pages", flush=True)
            break

        cursor = page_info["endCursor"]

        # Polite rate limiting: 0.5s between pages
        time.sleep(0.5)

    # Merge: new/updated PRs from `fetched` override existing, keep the rest
    merged: dict[int, dict[str, Any]] = dict(existing)  # start with existing
    for pr in fetched:
        merged[pr["number"]] = pr  # overwrite with fresh data

    result = sorted(merged.values(), key=lambda p: p["number"], reverse=True)
    return result, api_calls


def main() -> None:
    """Entry point."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("[WARN] GITHUB_TOKEN not set — will hit 60 req/hour unauthenticated limit", flush=True)

    print("[INFO] Loading cache...", flush=True)
    cache = _load_cache()
    last_fetch = cache.get("last_fetch")
    prior_count = len(cache.get("prs", []))
    print(f"[INFO] Cache has {prior_count} PRs. Last fetch: {last_fetch}", flush=True)

    print("[INFO] Fetching PRs from GitHub...", flush=True)
    fetch_start = datetime.now(timezone.utc)
    try:
        prs, api_calls = fetch_all_prs(token, cache)
    except Exception as exc:
        print(f"[FATAL] Fetch failed: {exc}", flush=True)
        sys.exit(1)

    cache["last_fetch"] = fetch_start.isoformat()
    cache["api_calls_used"] = cache.get("api_calls_used", 0) + api_calls
    cache["prs"] = prs

    print(f"[INFO] Saving {len(prs)} PRs to {CACHE_PATH}", flush=True)
    _save_cache(cache)

    print(
        f"[DONE] fetch_prs.py complete. "
        f"PRs: {len(prs)} | API calls this run: {api_calls} | "
        f"Total API calls: {cache['api_calls_used']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
