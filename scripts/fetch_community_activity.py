#!/usr/bin/env python3
"""Fetch community activity metrics from GitHub API for the Agora.

Produces data/community_activity.csv with daily counts:
  date, mato_comments, other_comments, prs_created, prs_updated
"""

import csv
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

OWNER = "openai"
REPO = "parameter-golf"
USER = "MatoTeziTanka"
START_DATE = "2026-03-18"
OUT_PATH = Path(__file__).parent.parent / "data" / "community_activity.csv"

# EST offset (UTC-4 during EDT)
EST_OFFSET = timedelta(hours=-4)


def gh_api(endpoint: str, params: dict | None = None) -> list | dict:
    """Call gh api and return parsed JSON."""
    cmd = ["gh", "api", endpoint, "--method", "GET"]
    for k, v in (params or {}).items():
        cmd += ["-f", f"{k}={v}"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"[WARN] gh api failed: {result.stderr[:200]}", file=sys.stderr)
        return []
    return json.loads(result.stdout)


def fetch_comments_by_day() -> dict[str, dict[str, int]]:
    """Fetch all issue comments, bucket by EST day and user."""
    by_day: dict[str, dict[str, int]] = defaultdict(lambda: {"mato": 0, "other": 0})
    page = 1
    since = f"{START_DATE}T00:00:00Z"

    while True:
        data = gh_api(
            f"repos/{OWNER}/{REPO}/issues/comments",
            {"sort": "created", "direction": "asc", "since": since, "per_page": "100", "page": str(page)},
        )
        if not data:
            break
        for comment in data:
            created = comment.get("created_at", "")
            if not created:
                continue
            utc_dt = datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            est_dt = utc_dt + EST_OFFSET
            day = est_dt.strftime("%Y-%m-%d")
            login = comment.get("user", {}).get("login", "")
            if login == USER:
                by_day[day]["mato"] += 1
            else:
                by_day[day]["other"] += 1
        if len(data) < 100:
            break
        page += 1
        print(f"  comments page {page}...", flush=True)

    return dict(by_day)


def fetch_prs_by_day() -> tuple[dict[str, int], dict[str, int]]:
    """Fetch PR created and updated counts by EST day."""
    created_by_day: dict[str, int] = defaultdict(int)
    updated_by_day: dict[str, int] = defaultdict(int)
    page = 1

    while True:
        data = gh_api(
            f"repos/{OWNER}/{REPO}/pulls",
            {"state": "all", "sort": "created", "direction": "asc", "per_page": "100", "page": str(page)},
        )
        if not data:
            break
        for pr in data:
            c = pr.get("created_at", "")
            u = pr.get("updated_at", "")
            if c:
                utc_dt = datetime.strptime(c, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                est_dt = utc_dt + EST_OFFSET
                day = est_dt.strftime("%Y-%m-%d")
                if day >= START_DATE:
                    created_by_day[day] += 1
            if u:
                utc_dt = datetime.strptime(u, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                est_dt = utc_dt + EST_OFFSET
                day = est_dt.strftime("%Y-%m-%d")
                if day >= START_DATE:
                    updated_by_day[day] += 1
        if len(data) < 100:
            break
        page += 1
        print(f"  PRs page {page}...", flush=True)

    return dict(created_by_day), dict(updated_by_day)


def main():
    print("[FETCH] Community activity data...", flush=True)

    print("  Fetching comments...", flush=True)
    comments = fetch_comments_by_day()

    print("  Fetching PRs...", flush=True)
    prs_created, prs_updated = fetch_prs_by_day()

    # Build date range
    start = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end = datetime.now(timezone.utc).date() + timedelta(days=0)
    all_dates = []
    d = start
    while d <= end:
        all_dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)

    # Write CSV
    with OUT_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "mato_comments", "other_comments", "prs_created", "prs_updated"])
        for day in all_dates:
            c = comments.get(day, {"mato": 0, "other": 0})
            writer.writerow([
                day,
                c["mato"],
                c["other"],
                prs_created.get(day, 0),
                prs_updated.get(day, 0),
            ])

    print(f"[FETCH] Wrote {len(all_dates)} days to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
