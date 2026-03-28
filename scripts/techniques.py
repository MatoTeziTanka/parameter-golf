#!/usr/bin/env python3
"""Build AGORA Phase 3 technique analytics from the PR cache."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = REPO_ROOT / "data" / "pr_cache.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "techniques.json"

TECHNIQUE_DEFS: tuple[dict[str, Any], ...] = (
    {"name": "LeakyReLU²", "aliases": (r"leaky\s*relu(?:²|2)?", r"leakyrelu(?:²|2)?"), "kind": "legal"},
    {"name": "SwiGLU", "aliases": (r"swiglu",), "kind": "legal"},
    {"name": "GeGLU", "aliases": (r"geglu",), "kind": "legal"},
    {"name": "MoE", "aliases": (r"\bmoe\b", r"mixture of experts"), "kind": "legal"},
    {"name": "SSM/Mamba/S4D", "aliases": (r"\bssm\b", r"mamba", r"s4d"), "kind": "legal"},
    {"name": "JEPA", "aliases": (r"\bjepa\b",), "kind": "legal"},
    {"name": "TTT", "aliases": (r"\bttt\b", r"test[- ]time training"), "kind": "legal"},
    {"name": "EMA", "aliases": (r"\bema\b", r"exponential moving average"), "kind": "legal"},
    {"name": "SWA", "aliases": (r"\bswa\b", r"stochastic weight averaging"), "kind": "legal"},
    {"name": "QAT", "aliases": (r"\bqat\b", r"quantization[- ]aware"), "kind": "legal"},
    {"name": "GPTQ", "aliases": (r"\bgptq\b",), "kind": "legal"},
    {"name": "Muon optimizer", "aliases": (r"\bmuon\b",), "kind": "legal"},
    {"name": "Parallel Muon", "aliases": (r"parallel[- ]muon",), "kind": "legal"},
    {"name": "Flash Attention", "aliases": (r"flash[-_ ]attention", r"flash[-_ ]attn"), "kind": "legal"},
    {"name": "BigramHash", "aliases": (r"bigramhash", r"bigram[- ]hash"), "kind": "legal"},
    {"name": "Value Residual", "aliases": (r"value[- ]residual",), "kind": "legal"},
    {"name": "XSA", "aliases": (r"\bxsa\b",), "kind": "legal"},
    {"name": "U-Net skips", "aliases": (r"u[- ]net skips?", r"unet skips?"), "kind": "legal"},
    {"name": "Differential Attention", "aliases": (r"differential attention",), "kind": "legal"},
    {"name": "Weight Sharing", "aliases": (r"weight sharing", r"shared weights?"), "kind": "legal"},
    {"name": "Progressive Depth", "aliases": (r"progressive depth",), "kind": "legal"},
    {"name": "Ternary Weights", "aliases": (r"ternary weights?", r"\bternary\b"), "kind": "legal"},
    {"name": "N-gram Cache", "aliases": (r"n[- ]gram", r"\bngram\b"), "kind": "banned"},
    {"name": "Two-Pass", "aliases": (r"two[- ]pass", r"2[- ]pass", r"rescoring", r"rescore"), "kind": "banned"},
)

TECHNIQUE_ORDER = tuple(item["name"] for item in TECHNIQUE_DEFS)
BANNED_TECHNIQUES = frozenset(item["name"] for item in TECHNIQUE_DEFS if item["kind"] == "banned")
NON_NEURAL_KEYWORDS = frozenset({"ngram", "ngram_path", "two_pass", "cache_eval", "cache_path", "prefill"})
STATUS_ORDER = ("ALIVE", "AT_RISK", "DEAD", "INCOMPLETE", "UNKNOWN")

_TECHNIQUE_PATTERNS = {
    item["name"]: [re.compile(pattern, re.IGNORECASE) for pattern in item["aliases"]]
    for item in TECHNIQUE_DEFS
}


def load_prs(cache_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    prs = payload.get("prs", []) if isinstance(payload, dict) else payload
    if not isinstance(prs, list):
        raise ValueError(f"Unexpected cache shape in {cache_path}")
    return prs


def _search_text(pr: dict[str, Any]) -> str:
    parts = [
        pr.get("title") or "",
        pr.get("body") or "",
        pr.get("technique_summary") or "",
        " ".join(pr.get("compliance_keywords") or []),
    ]
    return "\n".join(parts)


def match_techniques(pr: dict[str, Any]) -> list[str]:
    text = _search_text(pr)
    matches: list[str] = []
    for name in TECHNIQUE_ORDER:
        if any(pattern.search(text) for pattern in _TECHNIQUE_PATTERNS[name]):
            matches.append(name)
    return matches


def annotate_prs(prs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for pr in prs:
        record = dict(pr)
        record["matched_techniques"] = match_techniques(record)
        annotated.append(record)
    return annotated


def is_neural_only(pr: dict[str, Any]) -> bool:
    keywords = {keyword.lower() for keyword in pr.get("compliance_keywords") or []}
    flags = {flag.lower() for flag in pr.get("flags") or []}
    techniques = set(pr.get("matched_techniques") or [])
    if techniques & BANNED_TECHNIQUES:
        return False
    if keywords & NON_NEURAL_KEYWORDS:
        return False
    return not any("prefill" in flag for flag in flags)


def _positive_bpb(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    if value <= 0:
        return None
    return float(value)


def _pr_stub(pr: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": pr.get("number"),
        "author": pr.get("author"),
        "status": pr.get("status", "UNKNOWN"),
        "bpb": _positive_bpb(pr.get("bpb")),
        "created": pr.get("created"),
    }


def _sort_key_for_pr(pr: dict[str, Any]) -> tuple[float, int]:
    bpb = _positive_bpb(pr.get("bpb"))
    return (bpb if bpb is not None else float("inf"), int(pr.get("number") or 0))


def build_payload(prs: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = defaultdict(int)
    technique_entries: list[dict[str, Any]] = []
    co_occurrence: dict[tuple[str, str], int] = defaultdict(int)

    for pr in prs:
        status_counts[pr.get("status", "UNKNOWN")] += 1
        legal_matches = [name for name in pr.get("matched_techniques") or [] if name not in BANNED_TECHNIQUES]
        for index, left in enumerate(legal_matches):
            for right in legal_matches[index + 1:]:
                co_occurrence[tuple(sorted((left, right)))] += 1

    for definition in TECHNIQUE_DEFS:
        name = definition["name"]
        group = [pr for pr in prs if name in (pr.get("matched_techniques") or [])]
        authors: dict[str, list[dict[str, Any]]] = defaultdict(list)
        per_status: dict[str, int] = defaultdict(int)
        alive_with_bpb: list[dict[str, Any]] = []

        for pr in group:
            authors[pr.get("author") or "unknown"].append(pr)
            per_status[pr.get("status", "UNKNOWN")] += 1
            if pr.get("status") == "ALIVE" and _positive_bpb(pr.get("bpb")) is not None:
                alive_with_bpb.append(pr)

        best_alive: dict[str, Any] | None = None
        if alive_with_bpb:
            best_pr = min(alive_with_bpb, key=_sort_key_for_pr)
            best_alive = {
                "number": best_pr.get("number"),
                "author": best_pr.get("author"),
                "bpb": _positive_bpb(best_pr.get("bpb")),
            }

        people: list[dict[str, Any]] = []
        for author, author_prs in authors.items():
            author_prs_sorted = sorted(author_prs, key=lambda item: (_sort_key_for_pr(item), item.get("created") or ""))
            people.append(
                {
                    "author": author,
                    "pr_count": len(author_prs_sorted),
                    "best_bpb": next((_positive_bpb(item.get("bpb")) for item in author_prs_sorted if _positive_bpb(item.get("bpb")) is not None), None),
                    "prs": [_pr_stub(item) for item in sorted(author_prs_sorted, key=lambda item: int(item.get("number") or 0))],
                }
            )

        people.sort(
            key=lambda person: (
                -person["pr_count"],
                person["best_bpb"] if person["best_bpb"] is not None else float("inf"),
                person["author"].lower(),
            )
        )

        technique_entries.append(
            {
                "name": name,
                "kind": definition["kind"],
                "count": len(group),
                "author_count": len(authors),
                "alive_count": per_status.get("ALIVE", 0),
                "dead_count": per_status.get("DEAD", 0),
                "at_risk_count": per_status.get("AT_RISK", 0),
                "incomplete_count": per_status.get("INCOMPLETE", 0),
                "unknown_count": per_status.get("UNKNOWN", 0),
                "best_alive": best_alive,
                "people": people,
                "prs": [_pr_stub(pr) for pr in sorted(group, key=lambda item: int(item.get("number") or 0))],
            }
        )

    technique_entries.sort(key=lambda item: (item["kind"] == "banned", -item["count"], item["name"].lower()))

    counts_by_name = {item["name"]: item["count"] for item in technique_entries}
    author_counts_by_name = {item["name"]: item["author_count"] for item in technique_entries}
    untried_combinations: list[dict[str, Any]] = []
    legal_names = [item["name"] for item in technique_entries if item["kind"] == "legal" and item["count"] > 0]
    for index, left in enumerate(legal_names):
        for right in legal_names[index + 1:]:
            key = tuple(sorted((left, right)))
            if co_occurrence.get(key, 0) != 0:
                continue
            untried_combinations.append(
                {
                    "left": left,
                    "right": right,
                    "left_count": counts_by_name[left],
                    "right_count": counts_by_name[right],
                    "left_people": author_counts_by_name[left],
                    "right_people": author_counts_by_name[right],
                    "score": counts_by_name[left] * counts_by_name[right],
                }
            )

    untried_combinations.sort(
        key=lambda item: (
            -item["score"],
            -(item["left_people"] + item["right_people"]),
            item["left"].lower(),
            item["right"].lower(),
        )
    )

    untouched = [item["name"] for item in technique_entries if item["kind"] == "legal" and item["count"] == 0]

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_cache": str(DEFAULT_CACHE.relative_to(REPO_ROOT)),
        "summary": {
            "pr_count": len(prs),
            "matched_pr_count": sum(1 for pr in prs if pr.get("matched_techniques")),
            "technique_count": len(TECHNIQUE_DEFS),
            "untouched_techniques": untouched,
            "status_counts": {status: status_counts.get(status, 0) for status in STATUS_ORDER},
        },
        "techniques": technique_entries,
        "collaboration": sorted(
            technique_entries,
            key=lambda item: (-item["author_count"], -item["count"], item["name"].lower()),
        ),
        "untried_combinations": untried_combinations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_CACHE, help="Path to data/pr_cache.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to write data/techniques.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prs = annotate_prs(load_prs(args.input))
    payload = build_payload(prs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(
        f"[techniques] wrote {args.output} with {len(payload['techniques'])} techniques "
        f"from {payload['summary']['pr_count']} PRs",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
