#!/usr/bin/env python3
"""
fetch_community.py — Process AGORA community issue-form submissions.

Reads open issues from the MatoTeziTanka/parameter-golf fork via the GitHub
REST API, parses structured issue-form fields, appends new records to the
community JSON datasets under data/, logs site-facing changes to
data/site_changelog.json, posts an "Added to AGORA" comment, and closes the
processed issues.

Usage:
    GITHUB_TOKEN=<token> python scripts/fetch_community.py
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"

REPO_OWNER = "MatoTeziTanka"
REPO_NAME = "parameter-golf"
API_ROOT = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
TIMEOUT_SECONDS = 30
COMMENT_MARKER = "<!-- AGORA:PROCESSED -->"
COMMENT_BODY = (
    f"{COMMENT_MARKER}\n"
    "Added to AGORA. Thanks for contributing — this submission has been synced "
    "into the community data and the issue can now be closed."
)

FIELD_PATTERN = re.compile(r"^###\s+(.+?)\n+(.*?)(?=^###\s+|\Z)", re.MULTILINE | re.DOTALL)
MONEY_PATTERN = re.compile(r"[-+]?\$?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)")
PR_NUMBER_PATTERN = re.compile(r"(?:#|/pull/)(\d+)|\bPR\s*(\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class CollectionSpec:
    path: Path
    list_key: str


COLLECTION_SPECS: dict[str, CollectionSpec] = {
    "funding-report": CollectionSpec(DATA_DIR / "funding.json", "reports"),
    "correction": CollectionSpec(DATA_DIR / "corrections.json", "corrections"),
    "blocked-on-compute": CollectionSpec(
        DATA_DIR / "blocked_experiments.json",
        "experiments",
    ),
    "technique-suggestion": CollectionSpec(
        DATA_DIR / "community_techniques.json",
        "suggestions",
    ),
    "resource-suggestion": CollectionSpec(
        DATA_DIR / "community_resources.json",
        "resources",
    ),
    "general-feedback": CollectionSpec(
        DATA_DIR / "community_feedback.json",
        "feedback",
    ),
}

CHANGELOG_SPEC = CollectionSpec(DATA_DIR / "site_changelog.json", "entries")

TYPE_LABELS: dict[str, set[str]] = {
    "funding-report": {"funding-report", "community-funding-report"},
    "correction": {"correction", "pr-classification-correction"},
    "technique-suggestion": {"technique-suggestion"},
    "blocked-on-compute": {"blocked-on-compute", "blocked"},
    "resource-suggestion": {"resource-suggestion", "compute-resource-suggestion"},
    "general-feedback": {"general-feedback", "feedback"},
}

REQUIRED_FIELDS: dict[str, set[str]] = {
    "funding-report": {
        "github username",
        "grant amount",
        "self-funded amount",
        "platforms used",
        "proof link",
    },
    "correction": {"pr number", "current status", "what it should be", "evidence link"},
    "technique-suggestion": {"technique name", "description"},
    "blocked-on-compute": {
        "github username",
        "pr link",
        "what you want to test",
        "what you need",
        "how much spent",
    },
    "resource-suggestion": {"provider name", "gpu type", "pricing", "free tier details"},
    "general-feedback": {"subject", "description"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def normalize_field_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def normalize_no_response(value: str) -> str:
    cleaned = value.strip()
    if cleaned == "_No response_":
        return ""
    return cleaned


def parse_issue_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    if not body:
        return fields
    for match in FIELD_PATTERN.finditer(body):
        name = normalize_field_name(match.group(1))
        value = normalize_no_response(match.group(2))
        fields[name] = value
    return fields


def parse_checkbox_values(value: str) -> list[str]:
    selections: list[str] = []
    for line in value.splitlines():
        stripped = line.strip()
        if stripped.startswith("- [x] ") or stripped.startswith("- [X] "):
            selections.append(stripped[6:].strip())
    return selections


def parse_money(value: str) -> float | None:
    match = MONEY_PATTERN.search(value)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def parse_pr_number(value: str) -> int | None:
    match = PR_NUMBER_PATTERN.search(value)
    if not match:
        return None
    group = match.group(1) or match.group(2)
    return int(group) if group else None


def parse_pr_numbers(value: str) -> list[int]:
    numbers = {
        int(group)
        for match in PR_NUMBER_PATTERN.finditer(value)
        for group in match.groups()
        if group
    }
    return sorted(numbers)


def strip_at_prefix(username: str) -> str:
    return username.strip().lstrip("@")


def make_session(token: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "agora-community-fetcher",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    return session


def github_request(
    session: requests.Session,
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> Any:
    response = session.request(
        method,
        url,
        params=params,
        json=json_body,
        timeout=TIMEOUT_SECONDS,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text.strip()
        raise RuntimeError(f"GitHub API {method} {url} failed: {exc} :: {detail}") from exc
    if response.status_code == 204 or not response.content:
        return None
    return response.json()


def list_open_issues(session: requests.Session) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = github_request(
            session,
            "GET",
            f"{API_ROOT}/issues",
            params={"state": "open", "per_page": 100, "page": page},
        )
        if not payload:
            break
        for item in payload:
            if "pull_request" in item:
                continue
            issues.append(item)
        page += 1
    return issues


def list_issue_comments(session: requests.Session, issue_number: int) -> list[dict[str, Any]]:
    comments: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = github_request(
            session,
            "GET",
            f"{API_ROOT}/issues/{issue_number}/comments",
            params={"per_page": 100, "page": page},
        )
        if not payload:
            break
        comments.extend(payload)
        page += 1
    return comments


def ensure_issue_comment(session: requests.Session, issue_number: int) -> bool:
    comments = list_issue_comments(session, issue_number)
    if any(COMMENT_MARKER in (comment.get("body") or "") for comment in comments):
        return False
    github_request(
        session,
        "POST",
        f"{API_ROOT}/issues/{issue_number}/comments",
        json_body={"body": COMMENT_BODY},
    )
    return True


def close_issue(session: requests.Session, issue_number: int) -> None:
    github_request(
        session,
        "PATCH",
        f"{API_ROOT}/issues/{issue_number}",
        json_body={"state": "closed", "state_reason": "completed"},
    )


def default_collection(spec: CollectionSpec) -> dict[str, Any]:
    return {"last_updated": None, spec.list_key: []}


def load_collection(spec: CollectionSpec) -> tuple[dict[str, Any], bool]:
    if not spec.path.exists():
        return default_collection(spec), True
    with spec.path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected object in {spec.path}")
    if spec.list_key not in payload or not isinstance(payload[spec.list_key], list):
        raise RuntimeError(f"Expected list key '{spec.list_key}' in {spec.path}")
    payload.setdefault("last_updated", None)
    return payload, False


def save_collection(spec: CollectionSpec, payload: dict[str, Any]) -> None:
    spec.path.parent.mkdir(parents=True, exist_ok=True)
    with spec.path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def infer_issue_type(issue: dict[str, Any], fields: dict[str, str]) -> str | None:
    label_names = {normalize_label(label.get("name", "")) for label in issue.get("labels", [])}
    for issue_type, candidate_labels in TYPE_LABELS.items():
        if label_names & candidate_labels:
            return issue_type

    field_names = set(fields)
    for issue_type, required_fields in REQUIRED_FIELDS.items():
        if required_fields.issubset(field_names):
            return issue_type

    title = normalize_label(issue.get("title", ""))
    if "funding" in title:
        return "funding-report"
    if "correction" in title or "classification" in title:
        return "correction"
    if "blocked" in title:
        return "blocked-on-compute"
    if "resource" in title:
        return "resource-suggestion"
    if "technique" in title:
        return "technique-suggestion"
    if "feedback" in title:
        return "general-feedback"
    return None


def missing_required_fields(issue_type: str, fields: dict[str, str]) -> list[str]:
    return [
        field_name
        for field_name in sorted(REQUIRED_FIELDS[issue_type])
        if not normalize_no_response(fields.get(field_name, ""))
    ]


def build_funding_record(issue: dict[str, Any], fields: dict[str, str], processed_at: str) -> dict[str, Any]:
    platforms_raw = fields.get("platforms used", "")
    return {
        "source_issue_number": issue["number"],
        "source_issue_url": issue["html_url"],
        "issue_title": issue.get("title", ""),
        "issue_created_at": issue.get("created_at"),
        "processed_at": processed_at,
        "github_username": strip_at_prefix(fields["github username"]),
        "grant_amount": fields["grant amount"],
        "grant_amount_usd": parse_money(fields["grant amount"]),
        "self_funded_amount": fields["self-funded amount"],
        "self_funded_amount_usd": parse_money(fields["self-funded amount"]),
        "platforms_used": parse_checkbox_values(platforms_raw),
        "platforms_used_raw": platforms_raw,
        "proof_link": fields["proof link"],
    }


def build_correction_record(issue: dict[str, Any], fields: dict[str, str], processed_at: str) -> dict[str, Any]:
    return {
        "source_issue_number": issue["number"],
        "source_issue_url": issue["html_url"],
        "issue_title": issue.get("title", ""),
        "issue_created_at": issue.get("created_at"),
        "processed_at": processed_at,
        "pr_number": parse_pr_number(fields["pr number"]),
        "pr_number_raw": fields["pr number"],
        "current_status": fields["current status"],
        "suggested_status": fields["what it should be"],
        "evidence_link": fields["evidence link"],
    }


def build_blocked_record(issue: dict[str, Any], fields: dict[str, str], processed_at: str) -> dict[str, Any]:
    return {
        "source_issue_number": issue["number"],
        "source_issue_url": issue["html_url"],
        "issue_title": issue.get("title", ""),
        "issue_created_at": issue.get("created_at"),
        "processed_at": processed_at,
        "github_username": strip_at_prefix(fields["github username"]),
        "pr_link": fields["pr link"],
        "pr_number": parse_pr_number(fields["pr link"]),
        "what_you_want_to_test": fields["what you want to test"],
        "what_you_need": fields["what you need"],
        "how_much_spent": fields["how much spent"],
        "how_much_spent_usd": parse_money(fields["how much spent"]),
    }


def build_technique_record(issue: dict[str, Any], fields: dict[str, str], processed_at: str) -> dict[str, Any]:
    related_prs_raw = fields.get("related prs", "")
    return {
        "source_issue_number": issue["number"],
        "source_issue_url": issue["html_url"],
        "issue_title": issue.get("title", ""),
        "issue_created_at": issue.get("created_at"),
        "processed_at": processed_at,
        "technique_name": fields["technique name"],
        "description": fields["description"],
        "related_prs": parse_pr_numbers(related_prs_raw),
        "related_prs_raw": related_prs_raw,
    }


def build_resource_record(issue: dict[str, Any], fields: dict[str, str], processed_at: str) -> dict[str, Any]:
    return {
        "source_issue_number": issue["number"],
        "source_issue_url": issue["html_url"],
        "issue_title": issue.get("title", ""),
        "issue_created_at": issue.get("created_at"),
        "processed_at": processed_at,
        "provider_name": fields["provider name"],
        "gpu_type": fields["gpu type"],
        "pricing": fields["pricing"],
        "free_tier_details": fields["free tier details"],
    }


def build_feedback_record(issue: dict[str, Any], fields: dict[str, str], processed_at: str) -> dict[str, Any]:
    return {
        "source_issue_number": issue["number"],
        "source_issue_url": issue["html_url"],
        "issue_title": issue.get("title", ""),
        "issue_created_at": issue.get("created_at"),
        "processed_at": processed_at,
        "subject": fields["subject"],
        "description": fields["description"],
    }


RecordBuilder = Callable[[dict[str, Any], dict[str, str], str], dict[str, Any]]

RECORD_BUILDERS: dict[str, RecordBuilder] = {
    "funding-report": build_funding_record,
    "correction": build_correction_record,
    "blocked-on-compute": build_blocked_record,
    "technique-suggestion": build_technique_record,
    "resource-suggestion": build_resource_record,
    "general-feedback": build_feedback_record,
}

CHANGE_DESCRIPTIONS: dict[str, str] = {
    "funding-report": "Added funding report",
    "correction": "Added PR classification correction",
    "blocked-on-compute": "Added blocked experiment report",
    "technique-suggestion": "Added community technique suggestion",
    "resource-suggestion": "Added compute resource suggestion",
    "general-feedback": "Added general feedback submission",
}


def record_exists(payload: dict[str, Any], list_key: str, issue_number: int) -> bool:
    return any(item.get("source_issue_number") == issue_number for item in payload[list_key])


def append_unique_record(payload: dict[str, Any], list_key: str, record: dict[str, Any]) -> bool:
    issue_number = record["source_issue_number"]
    if record_exists(payload, list_key, issue_number):
        return False
    payload[list_key].append(record)
    payload[list_key].sort(key=lambda item: item.get("source_issue_number", 0))
    return True


def append_changelog_entry(
    changelog: dict[str, Any],
    *,
    timestamp: str,
    issue_type: str,
    issue_number: int,
    description: str,
) -> None:
    changelog[CHANGELOG_SPEC.list_key].append(
        {
            "timestamp": timestamp,
            "type": issue_type,
            "issue_number": issue_number,
            "description": description,
        }
    )


def ensure_default_files(collections: dict[str, dict[str, Any]], missing: dict[str, bool]) -> None:
    for issue_type, spec in COLLECTION_SPECS.items():
        if missing[issue_type]:
            collections[issue_type]["last_updated"] = utc_now()
            save_collection(spec, collections[issue_type])


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("[WARN] GITHUB_TOKEN not set — skipping community issue fetch", flush=True)
        return 0

    session = make_session(token)

    collections: dict[str, dict[str, Any]] = {}
    missing_files: dict[str, bool] = {}
    for issue_type, spec in COLLECTION_SPECS.items():
        payload, missing = load_collection(spec)
        collections[issue_type] = payload
        missing_files[issue_type] = missing

    changelog, changelog_missing = load_collection(CHANGELOG_SPEC)
    if changelog_missing:
        changelog["last_updated"] = utc_now()
        save_collection(CHANGELOG_SPEC, changelog)

    print(f"[INFO] Fetching open community issues from {REPO_OWNER}/{REPO_NAME}", flush=True)
    issues = list_open_issues(session)
    print(f"[INFO] Found {len(issues)} open issues", flush=True)

    processed_count = 0
    added_count = 0
    skipped_count = 0

    for issue in issues:
        issue_number = issue["number"]
        fields = parse_issue_fields(issue.get("body") or "")
        issue_type = infer_issue_type(issue, fields)

        if issue_type is None:
            print(f"[SKIP] #{issue_number} has no recognized community issue form", flush=True)
            skipped_count += 1
            continue

        missing_fields = missing_required_fields(issue_type, fields)
        if missing_fields:
            print(
                f"[SKIP] #{issue_number} missing required fields: {', '.join(missing_fields)}",
                flush=True,
            )
            skipped_count += 1
            continue

        processed_at = utc_now()
        spec = COLLECTION_SPECS[issue_type]
        payload = collections[issue_type]
        record = RECORD_BUILDERS[issue_type](issue, fields, processed_at)

        added = append_unique_record(payload, spec.list_key, record)
        if added:
            payload["last_updated"] = processed_at
            save_collection(spec, payload)
            append_changelog_entry(
                changelog,
                timestamp=processed_at,
                issue_type=issue_type,
                issue_number=issue_number,
                description=f"{CHANGE_DESCRIPTIONS[issue_type]} from issue #{issue_number}.",
            )
            changelog["last_updated"] = processed_at
            save_collection(CHANGELOG_SPEC, changelog)
            added_count += 1
            missing_files[issue_type] = False
            print(f"[ADD] #{issue_number} recorded as {issue_type}", flush=True)
        else:
            print(f"[INFO] #{issue_number} already recorded locally; ensuring close-out", flush=True)

        comment_added = ensure_issue_comment(session, issue_number)
        close_issue(session, issue_number)
        processed_count += 1
        print(
            f"[DONE] #{issue_number} processed ({'commented' if comment_added else 'comment already present'}, closed)",
            flush=True,
        )

    ensure_default_files(collections, missing_files)

    print(
        f"[DONE] fetch_community.py complete. Processed: {processed_count} | "
        f"Added: {added_count} | Skipped: {skipped_count}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
