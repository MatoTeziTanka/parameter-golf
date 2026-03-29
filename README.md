# OLYMPUS: The Agora

**Community companion for [OpenAI Parameter Golf](https://github.com/openai/parameter-golf)**

**[View the live site](https://matotezitanka.github.io/parameter-golf/)**

---

## What is The Agora?

Parameter Golf has 1000+ PRs, rules that change mid-game, and $1M in promised compute that nobody can track. The Agora answers the questions the competition doesn't:

1. **"Is my submission legal?"** — Plain-English checklist + rule change history
2. **"Is my submission alive?"** — Dual leaderboard: compliant-only and full archive
3. **"What should I try next?"** — Technique map with 24 tracked approaches, alive/dead ratios
4. **"Who else is working on this?"** — Collaboration finder, untried combinations
5. **"How is the competition evolving?"** — BPB timeline with the March 27 cliff
6. **"How much is the community spending?"** — Funding transparency tracker
7. **"Am I getting good value?"** — Serverless vs pods cost comparison
8. **"What bugs should I know about?"** — Community-reported issues that affect scores

## Features

- **Live Leaderboard** — 996 PRs auto-classified (ALIVE / DEAD / AT-RISK / INCOMPLETE)
- **Compliance Engine** — Flags banned techniques: n-gram caches, two-pass rescoring, eval-time GPTQ
- **Technique Map** — 24 techniques tracked, grouped by PRs, best BPB per technique
- **BPB Timeline** — SOTA over time, key events, the March 27 cliff
- **Funding Tracker** — Community-reported compute spending and grant distribution
- **Blocked on Compute Board** — Researchers with ideas but no GPU access
- **Compute Survival Guide** — Why serverless beats pods (we burned $242 learning this)
- **Bug Alerts** — BPB underestimation bug ([#897](https://github.com/openai/parameter-golf/issues/897)), INT6 scale clamp ([#775](https://github.com/openai/parameter-golf/issues/775))
- **Community Input** — 6 issue templates for corrections, funding reports, technique suggestions

## Community

- [Submit feedback or corrections](https://github.com/MatoTeziTanka/parameter-golf/issues/new/choose)
- [Join the discussion](https://github.com/MatoTeziTanka/parameter-golf/discussions)
- [Star the repo](https://github.com/MatoTeziTanka/parameter-golf) to get updates

## How It Works

```
scripts/fetch_prs.py       — GraphQL scraper (openai/parameter-golf)
scripts/fetch_community.py — Ingest issues from our fork
scripts/classify.py        — Compliance classification engine
scripts/generate_site.py   — Single-page HTML generator
scripts/run_pipeline.py    — Orchestrator: fetch -> classify -> generate

.github/workflows/update-site.yml  — Cron every 30 min
.github/ISSUE_TEMPLATE/             — 6 structured templates
```

Data sources: GitHub GraphQL API, Issue #140 (community commentary), Issue #942 (funding reports), PR diffs (submission.json parsing).

## Transparency

Built by [@MatoTeziTanka](https://github.com/MatoTeziTanka) ([Light Speed Up](https://lightspeedup.com)), an active participant. Our [PR #769](https://github.com/openai/parameter-golf/pull/769) (0.8495 BPB) was closed in the March 27 ruling. We are not neutral — all classifications are automated from public GitHub data and disputable via [Issues](https://github.com/MatoTeziTanka/parameter-golf/issues).

Special thanks to [@ddeturk24](https://github.com/ddeturk24) — OLYMPUS collaborator and the person who made it all possible.

## Versions

| Version | Date | What shipped |
|---------|------|-------------|
| v0.5.2 | 2026-03-29 | Community bug alerts (BPB bug, INT6 clamp) |
| v0.5.1 | 2026-03-29 | UX overhaul: scrollable tables, nav fix, neural filter, ALIVE-first sort |
| v0.5.0 | 2026-03-28 | GitHub Actions cron, 6 issue templates, community ingestion |
| v0.4.0 | 2026-03-28 | Community input pipeline |
| v0.3.0 | 2026-03-28 | Technique map, BPB timeline, collaboration finder |
| v0.2.0 | 2026-03-28 | Live leaderboard (996 PRs via GraphQL) |
| v0.1.0 | 2026-03-28 | Static foundation: checklist, funding tracker, compute guide |

## License

MIT — same as the upstream repo.
