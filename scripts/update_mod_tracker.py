#!/usr/bin/env python3
"""Update the OpenAI maintainer activity tracker on the Agora.

Reads the validated mod list, queries GitHub Events API for each, computes
days-silent, and writes a JSON snapshot to data/mod_tracker.json. Optionally
patches the inline section in index.html for static rendering.

Usage:
  python3 scripts/update_mod_tracker.py            # write JSON only
  python3 scripts/update_mod_tracker.py --patch    # write JSON + update index.html
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Validated maintainers — confirmed merge/close authority on openai/parameter-golf
# Validation: author_association on closed PRs, OAI account suffix, closing
# another user's PR, repo collaborator status.
VALIDATED_MODS = [
    {
        "handle": "notapplica",
        "real_name": None,
        "role": "ran #140 commentary agent",
        "authority": "Closed Issue #140 (own bot's issue). CONTRIBUTOR.",
        "account_created": "2025-08-25",
    },
    {
        "handle": "valerio-oai",
        "real_name": None,
        "role": "rule rulings, PR closures",
        "authority": "Closed PR #1019 (someone else's record). Made March 27 mass-closure rulings on #677.",
        "account_created": "2026-02-10",
        "purpose_built": True,
        "pending": "Said 'considering options' on #677 about eval-time cache rulings",
    },
    {
        "handle": "0hq",
        "real_name": "Will DePue",
        "role": "competition founder",
        "authority": "COLLABORATOR. Repo founder. Opened #677 Illegal Submissions Megathread.",
        "account_created": "2017-08-02",
    },
    {
        "handle": "yuzhougu-oai",
        "real_name": None,
        "role": "PR merging",
        "authority": "CONTRIBUTOR. Has merged PRs on the repo.",
        "account_created": "2025-12-18",
    },
]

REPO_FILTER = "openai/parameter-golf"
EVENTS_URL = "https://api.github.com/users/{handle}/events?per_page=30"


def fetch_events(handle: str) -> list[dict]:
    """Fetch public events for a GitHub user."""
    req = urllib.request.Request(
        EVENTS_URL.format(handle=handle),
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "agora-mod-tracker",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  WARN: failed to fetch {handle}: {e}", file=sys.stderr)
        return []


def latest_repo_event(events: list[dict]) -> dict | None:
    """Find the most recent event in the parameter-golf repo."""
    for e in events:
        if e.get("repo", {}).get("name") == REPO_FILTER:
            return e
    return None


def event_summary(event: dict) -> str:
    """Build a human-readable summary of an event."""
    etype = event.get("type", "?")
    payload = event.get("payload", {})
    if etype == "IssueCommentEvent":
        issue = payload.get("issue", {}).get("number")
        return f"Comment on #{issue}"
    if etype == "IssuesEvent":
        action = payload.get("action", "?")
        issue = payload.get("issue", {}).get("number")
        return f"Issue #{issue} {action}"
    if etype == "PullRequestEvent":
        action = payload.get("action", "?")
        pr = payload.get("pull_request", {}).get("number")
        return f"PR #{pr} {action}"
    if etype == "PushEvent":
        return f"Push ({payload.get('size', '?')} commits)"
    if etype == "DeleteEvent":
        return f"Delete {payload.get('ref_type', '?')}"
    return etype


def days_silent(timestamp: str) -> float:
    """Compute days since the given ISO timestamp."""
    t = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    return (now - t).total_seconds() / 86400.0


def severity(days: float) -> str:
    """Color severity bucket for the UI."""
    if days < 1:
        return "active"  # green/yellow
    if days < 3:
        return "warning"  # yellow
    if days < 7:
        return "alert"  # orange
    return "silent"  # red


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", action="store_true", help="Patch index.html")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "mod_tracker.json",
    )
    args = parser.parse_args()

    snapshot = {
        "snapshot_at": datetime.now(timezone.utc).isoformat(),
        "repo": REPO_FILTER,
        "mods": [],
    }

    for mod in VALIDATED_MODS:
        handle = mod["handle"]
        print(f"Fetching {handle}...")
        events = fetch_events(handle)
        latest = latest_repo_event(events)

        entry = dict(mod)
        if latest:
            entry["last_action_at"] = latest["created_at"]
            entry["last_action_summary"] = event_summary(latest)
            entry["days_silent"] = round(days_silent(latest["created_at"]), 2)
            entry["severity"] = severity(entry["days_silent"])
        else:
            entry["last_action_at"] = None
            entry["last_action_summary"] = "No recent activity in repo"
            entry["days_silent"] = None
            entry["severity"] = "silent"

        snapshot["mods"].append(entry)
        print(
            f"  {handle}: {entry.get('last_action_at')} ({entry.get('days_silent')}d) [{entry.get('severity')}]"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"\nWrote {args.out}")

    if args.patch:
        # Future enhancement: regenerate the inline HTML section from snapshot.
        # For now the section is hand-edited; this just confirms the data is fresh.
        print("--patch not yet implemented (section is currently hand-rendered)")


if __name__ == "__main__":
    main()
