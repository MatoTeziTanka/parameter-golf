# Parameter-Golf — Contest Recap

**Published: 2026-05-09**
**Status: Contest closed for active submission. Agora continues as community-research archive.**

> The OpenAI Parameter Golf competition (Mar 18 to Apr 30, 2026) reached its frontier on **2026-04-29** when [PR #1626](https://github.com/openai/parameter-golf/pull/1626) by [@cocohearts](https://github.com/cocohearts) was merged at **val_bpb 1.07193** (3-seed mean, head SHA `9c5a57920101a7562a34e1500d78fd38b3b8355c`, technique: multi-phase global SGD + phased TTT, original PR author @dexhunter). This recap captures the contest's closure, what we (Light Speed Up) shipped, what we staged but never fired, and what survives forward.

---

## 1. Final frontier

| # | val_bpb | PR | Author | Technique | Status |
|---|--------:|----|--------|-----------|--------|
| **1** | **1.07193** | [#1626](https://github.com/openai/parameter-golf/pull/1626) | @cocohearts | Multi-phase global SGD + phased TTT | **MERGED 2026-04-29** |
| 2 | 1.07280 | [#1626](https://github.com/openai/parameter-golf/pull/1626) | @dexhunter (orig) | (seed-42 single-result reference) | merged via #1626 |
| 3 | 1.0810 | [#1493](https://github.com/openai/parameter-golf/pull/1493) | @bigbag | SP8192 + 3-layer recurrence + parallel residuals + legal TTT | merged earlier |

The full leaderboard is mirrored on this Agora — see the `#leaderboard` anchor on the index page.

## 2. What Light Speed Up shipped (publishings)

### Submissions (PRs to openai/parameter-golf)

- **PR #769** — PROTEUS+STYX (LeakyReLU(0.9)² + 5-gram eval cache + sliding window) at val_bpb 0.8495 (3-seed mean, std 0.0013). Closed in the 2026-03-27 ruling that re-classified target-token-in-eval-cache patterns as illegal across the n-gram family. Detailed audit + transparent superseded-results disclosure preserved on the PR thread.

### Community-research artifacts (this Agora)

- **[Dead-End Map (2026-04-15)](dead-end-map-2026-04-15.md)** — 14 techniques tested at the 16MB budget that didn't work, with deltas, rationale, and revisit conditions. Covers: INT4/RVQ/INT5+GPTQ quantization dead-ends, SWA/XSA attention variants, LeakyReLU 0.9 x parallel residuals path-dependent failure, EngramLite (cross-confirmed by Ciprian-Florin Ifrim's 2026-04-14 Discord post), LoopFormer, Scylla/Gravity tokenizer byte-accounting bugs, GDN kernel swap decision, plus compliance-illegal patterns (n-gram family-bug cluster, SLOT under #1336).

- **Compliance + funding-transparency dashboards** — Issue #140 cross-references, ruling timeline, dual leaderboard, funding-source tracking ($2,179 grants + $5,757+ self-funded across 8 participants), Compute Survival Guide.

- **Community impact analytics** — daily activity chart (community comments, Agora reviews, new PRs, PR updates) covering the contest window, with the +1941% PR-update spike during the 771-PR compliance sweep visible in the data.

- **Community Tools section** — links to Bortlesboat's three parameter-golf tools (parameter-golf-runpod-starter, parameter-golf-sweeps, parameter-golf-size-checker).

### Dataset republish (HuggingFace)

- **[LightSpeedUp/parameter-golf-data](https://huggingface.co/datasets/LightSpeedUp/parameter-golf-data)** — public mirror of the contest's tokenized FineWeb shards in seven variants:
  - `fineweb_sp1024/` (existing, mirrored)
  - `fineweb_sp4096/` (new, addresses Discord ask from @dém)
  - `fineweb_sp8192/` (new)
  - `fineweb_sp12288/` (new)
  - `fineweb_sp16384/` (new)
  - `fineweb_scylla_v2/` (new, fixes the 998-token byte-counting bug per PR #1314 / Issue #897 — corrected 1254-token vocab, byte-exact full-val audit)
  - `fineweb_scylla/` (legacy, deprecated, kept for reproducibility)

- **R2 mirror endpoint:** `pgolf-api.lightspeedup.com` (Cloudflare R2 backend).

### Compute access

- **[parameter-golf-private/docker/](https://github.com/MatoTeziTanka/parameter-golf-private/tree/main/docker)** Docker image (PyTorch 2.11.0 + cu128 + FA3 + Triton + sentencepiece + zstandard, validated for Hopper-class GPUs). Three bug fixes shipped during the contest after community testing surfaced gaps.

- **RunPod starter scripts** — open-source launch templates for the contest's standard 8xH100 SXM rung, contributed back to the community.

### Peer review

- 9 community reviews posted on round 1 (2026-03-26): PRs #846, #828, #764, #806, #808, #825, #779, #852, #769.
- 20 Tier-1 reviews posted on round 2 (2026-04-11) including the n-gram family-bug cluster ruling synthesis (10 downstream PRs closed, 1 ruled upstream by @valerio-oai on PR #779).
- 771-PR compliance sweep (2026-04-11 to 2026-04-12) surfacing the SLOT cluster and clean MERGE candidates (#1420, #1450).
- Issue #140 community toolkit post sharing the Docker image, RunPod template, scripts, bug fixes, peer-review summary.

## 3. What we staged but never fired

A planned 3-phase reproduction of PR #1626 was staged on 2026-05-09 (the same day the contest-closure pivot happened). The staging is preserved in olympus at `09_ARCHIVE/benchmarks/parameter-golf-pr1626-2026-05-09-contest-over/`:

- `POD-SPEC-2026-05-09.md` — full spec for both rung 2b ($2 1xH100 smoke) and rung 3 (~$13 8xH100 baseline reproduce)
- `PRE-FLIGHT-CHECKLIST-2026-05-09.md` — Mato pre-launch gates
- `RUNPOD-LAUNCH-2b-pr1626-smoke.sh` and `RUNPOD-LAUNCH-3-pr1626-baseline.sh` — launch scripts
- `runs/rung-2b-smoke-2026-05-09/` — actual evidence from the only attempted firing (3 crashes, ~$1.13 burn, $0 of useful work — root cause was the auto-shutdown env-injection bug, now tracked as Titan structural follow-up OT-20260509-008)

**Lessons preserved for future GPU work:**

1. **FA3 wheel resolution on Hopper** — the `flash_attn_3` wheel from `windreamer.github.io/flash-attention3-wheels/cu128_torch291` works on H100 (compute_cap 9.0). pip falls back silently to FA2 if the index is unreachable, then PR #1626's hard `from flash_attn_interface import ...` crashes at runtime. Pre-flight scripts should check this explicitly.

2. **Triton 3.5.1 + `python -c "..."` is incompatible** — `inspect.getsourcelines()` fails on stdin source. Always extract Triton kernels into a real `.py` file before invoking.

3. **`fineweb10B_sp8192` is NOT on the public HF mirror** — only `fineweb10B_sp1024` is in `willdepueoai/parameter-golf`. The sp8192 variant lives on the private `LightSpeedUp/parameter-golf-data` mirror and requires an HF token. Future PR #1626 reproductions need the token resolved up front.

4. **RunPod auto-shutdown trap requires env injection** — `POD_ID` and `RUNPOD_API_KEY` must be passed explicitly via `runpodctl pod create --env`, otherwise the trap fires but the pod stays running.

## 4. What survives forward (independent of contest closure)

- **STYX patent #63/975,190** — provisional, filed 2026-02-03, expires 2026-02-03. Patent track is independent of the parameter-golf contest. STYX-informed experiments continue under their own scope.

- **PYTHIA benchmark on FineWeb SP1024** — 3-seed BPB on the same dataset family remains a valid PYTHIA milestone for patent-evidence + this Agora's dead-end map, regardless of contest status.

- **Reproduction Verification Suite** — third-party reproduction of merged contest records remains community-research valuable as Agora content.

- **Frontier Analysis** — public technique-evolution study across 1600+ contest PRs is a lasting historical artifact.

- **This Agora itself** — community-research archive continues. Funding transparency, dead-end map, ruling timeline, leaderboard mirror all stay live. Future contests may follow; the Agora pattern (dual leaderboard + ruling history + dead-end map + funding transparency + compute survival guide) is the durable contribution.

## 5. Acknowledgements

To the 1600+ PR submitters who pushed the frontier from the initial 1.1194 baseline down to 1.07193 over six weeks. To @0hq and @valerio-oai for the rulings that kept compliance interpretable. To @cocohearts for the merging shepherd work on the final frontier PR. To @dexhunter for the technique. To @Bortlesboat for the open-source tooling. To @dém and the Discord community for the questions that drove our dataset republish. To Ciprian-Florin Ifrim for the documentation high-water mark on PR #1388. To everyone who let us peer-review their PRs adversarially without taking offense.

## 6. Get in touch

- Issues / corrections / feedback: [github.com/MatoTeziTanka/parameter-golf/issues](https://github.com/MatoTeziTanka/parameter-golf/issues)
- Submit a dead-end: PR to `docs/dead-end-map-2026-04-15.md`
- Submit a tool: see "Community Tools" on the main Agora index

---

**Related artifacts:**

- Olympus archive: `09_ARCHIVE/benchmarks/parameter-golf-pr1626-2026-05-09-contest-over/ARCHIVE-NOTE.md`
- Olympus closure commit: `2b15818` (7 OTs cancelled), `79d6717` (staging archived)
- Post-mortem of the rung-2b incident: `04_MOAT/audit-trail/runpod-rung2b-postmortem-2026-05-09.md`
- Titan structural follow-up: `08_FLOW/tasks/active/OT-20260509-008.md` (RunPod auto-shutdown env-injection fix)
