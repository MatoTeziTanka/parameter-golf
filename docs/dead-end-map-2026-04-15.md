# Parameter-Golf Dead-End Map

**Published: 2026-04-15**
**Compiled from:** PROTEUS experimentation logs (v6-v16), the 771-PR compliance sweep (Apr 11-12), and public ruling history on Issue #677 / #1336.

> A community gift. Every entry below is a technique someone (us or others) tested at the 16MB budget, hit a wall, and paid for in GPU time. Publishing this saves you from re-deriving the same results. Community contributions welcome — submit your own dead-ends via PR to this document.

## About this document

Parameter-Golf is a 16MB-artifact budget competition. Techniques that work at 1B+ parameter scale often fail here because the budget crowds them out. This map documents 14 techniques that **don't work at this budget** with the rationale, so you can spend your compute on things that might.

Format per entry:
- **What was tried**
- **Result** (delta vs baseline where captured)
- **Why it failed** (hypothesis, not always definitive)
- **When to revisit** (conditions under which this might become viable)
- **References** (PRs, rulings, memos)

Confidence markers:
- `[V]` verified with concrete evidence
- `[D]` documented in our own logs
- `[I]` inferred from code review, not bench-confirmed
- `[A]` assumed, not fully documented

---

## Architectural / Quantization

### 1. INT4 on transformer blocks `[D]`
- **What:** Full INT4 quantization-aware training on all transformer weight tensors.
- **Result:** **+3.73 BPB** — catastrophic.
- **Why it failed:** Gradient flow collapsed; activations clipped too aggressively for 11L @ 512d.
- **Revisit:** Never at this budget.
- **Ref:** PROTEUS V1 dead-end map.

### 2. RVQ INT4+INT4 codebook pairs `[D]`
- **What:** Residual Vector Quantization with 4-bit first + 4-bit second codebook on embeddings + weight tables.
- **Result:** 19.6× MSE vs full precision.
- **Why it failed:** Cascade compression lost too much information before second stage could refine.
- **Revisit:** Never with 2 codebooks. RVQ3+ unproven at this budget.

### 3. RVQ INT6+INT6 codebook pairs `[D]`
- **What:** INT6+INT6 dual codebook as companion test to the INT4 pair.
- **Result:** No savings — identical to pure INT6 storage.
- **Why it failed:** RVQ routing + sync overhead offset any compression benefit.
- **Revisit:** Never; indicates RVQ overhead is too high at 16MB.

### 4. INT5 + GPTQ double-quantization stack `[D]`
- **What:** INT5 pre-quantize during training, GPTQ int8 post-hoc for final artifact.
- **Result:** **+0.082 BPB** regression.
- **Why it failed:** GPTQ Hessian estimates built on INT5-corrupted activations; rounding error cascade.
- **Revisit:** Never as a stacked path. GPTQ directly on fp32-trained weights remains viable.

### 5. Low-rank SVD delta (Hadamard basis) `[D]`
- **What:** Low-rank SVD compression on residual pathway deltas; Hadamard basis.
- **Result:** ~0 recovery (white-noise reconstruction).
- **Why it failed:** Learned SVD basis was orthogonal to actual residual structure; Hadamard too generic for our model's residual variance.
- **Revisit:** Never with this basis. Basis-selection literature specific to small transformers not yet available.

---

## Attention / Architecture

### 6. SWA alone (no auxiliary heads, no XSA) `[D]`
- **What:** Stochastic Weight Averaging with EMA decay=0.95, cycle scheduling; no XSA or auxiliary heads.
- **Result:** **+0.0018 BPB** regression.
- **Why it failed:** SWA interacts poorly with Muon optimizer; EMA momentum conflicts with adaptive learning rates.
- **Revisit:** Possibly paired with auxiliary decoder; condition on aux-head working first.

### 7. SWA + XSA=4 (stochastic + 4 auxiliary heads) `[D]`
- **What:** SWA + XSA-4 (4 auxiliary prediction heads) full pipeline.
- **Result:** **+0.0031 BPB** regression.
- **Why it failed:** Auxiliary heads compete for gradient signal; effective per-head batch size shrinks.
- **Revisit:** Drop both components; stick to pure-neural tuning.

### 8. LeakyReLU slope 0.9 + parallel residuals `[D]`
- **What:** LeakyReLU(0.9) activation on all transformer blocks paired with parallel residuals from layer 7.
- **Result:** **+0.0054 BPB** penalty.
- **Why it failed:** Parallel residuals rewire gradient flow; steep 0.9 negative slope amplifies dead-ReLU regions. Parallel-residual lanes prefer 0.5 slope. Path-dependent interaction.
- **Revisit:** Only if parallel residuals removed. Frontier consistently uses 0.5² (squared LeakyReLU at 0.5).

### 9. EngramLite (multi-head hash n-gram embeddings) `[I]`
- **What:** Multi-order (bigram+trigram) hash-based embeddings with learned gating; replaces BigramHash.
- **Result:** Net negative BPB (exact delta not captured in our logs).
- **Why it failed:** At 11L @ 512d budget, n-gram auxiliary dims steal parameters from transformer; pure-neural beats hybrid.
- **Revisit:** Only if vocab scales ≥ 12288 AND n-gram becomes primary (not auxiliary).
- **External confirmation:** Ciprian-Florin Ifrim independently tested engram variants, posted on Discord 2026-04-14: *"gives about a 1% improvement + needs a lot of fiddling to get the right size (5% of model params from my research) as at this model size the engram will make the model 'stupid' by having it use the engram as a memory bank."* Matches our finding.
- **Separate research direction:** DeepSeek Engram (arxiv 2601.07372, Jan 2026) works at 27B+ scale — different regime, not applicable here.

### 10. LoopFormer (looped-layer fusion with learned attention gate) `[I]`
- **What:** Fused residual loops (layers 4-6 repeated) with learned attention gate at merge point.
- **Result:** Negative BPB impact (exact delta not recorded).
- **Why it failed:** Learnable gate became a bottleneck; repeated layers without new information = redundancy penalty.
- **Revisit:** Never at this budget. Mini Depth Recurrence (the merged PR #1204 variant) works because the gate is *discrete, not learned*.

---

## Tokenizers

### 11. Custom tokenizer: Scylla (998-token vocab, buggy) `[V]`
- **What:** Scylla 998-token vocab via standard `build_sentencepiece_luts()`.
- **Result:** **+4-6% BPB inflation** (false savings were actually a byte-accounting bug).
- **Why it failed:** Missing standalone U+2581 (word-boundary marker) caused fallback to byte-level encoding, counting 3 bytes instead of 1.
- **Revisit:** Use only PR #1314's corrected 1254-token vocab.
- **Ref:** Issue #897, PR #1143 (buggy original), PR #1314 (fix). Our PR #1289 was withdrawn over this bug. Full forensics in `SOW_HF_DATASET_REPUBLISH.md`.

### 12. Custom tokenizer: Gravity (BPE variant, unverified accounting) `[A]`
- **What:** Custom BPE-derived tokenizer, 1024-token base.
- **Result:** Byte-accounting inflation suspected, parallel to Scylla (exact delta not captured).
- **Why it failed:** Same root cause risk — custom byte-length LUTs not validated against reference encoder.
- **Revisit:** Never without third-party audit of byte accounting.

---

## Attention kernel / architecture swaps

### 13. GDN (Gated DeltaNet, swap for FlashAttention-3) `[D]`
- **What:** Decision memo 2026-04-06 evaluated replacing FA-3 softmax attention with FLA GDN kernels.
- **Result:** **Not tested; decided NO-GO.**
- **Why rejected:** Per-step wallclock slower (~90 ms vs ~88 ms on H100, 11L @ 512d). Step budget shrinks from ~6,800 → ~5,900-3,400. BPB regression expected at the timed track.
- **Revisit:** Only if an untimed track opens OR FLA publishes kernel that beats FA-3 at d=512/6h.
- **Ref:** `gdn-should-we-bet-memo.md` (2026-04-06) — detailed comparison table + 6 trigger conditions for revisiting.

---

## Compliance-illegal patterns (DO NOT ship)

### 14. N-gram eval-cache family bug — target-token leak `[V]`
- **What:** N-gram eval cache with target token hashed into lookup key:
  ```python
  full_key = ((ctx_hash ^ (target * primes[k])) & mask)
  ```
- **Result:** Illegal — violates eval causality. `p_t` becomes a function of `x_t` instead of prefix only.
- **Why it failed:** Evaluator can memorize correct output instead of predicting.
- **Revisit:** Never in this form.
- **Ref:** Ruled illegal by @valerio-oai on **PR #779** ([comment 4145781641](https://github.com/openai/parameter-golf/pull/779#issuecomment-4145781641), 2026-03-27). Downstream PRs closed under this ruling: #770, #798, #808, #825, #786, #797, #909, #940, #761, #764. Upstream parent PRs #659, #702, #727 by @lukacf still pending audit.

### 15. SLOT per-window delta + logit_bias optimization `[V]`
- **What:** Per-window `delta` + `logit_bias` optimized N AdamW steps against `(per_token_nll * mask).sum() / valid_count` where `mask` covers scored positions `[s:wlen]`, then scoring on same slice.
- **Result:** Compliance concern — optimizer minimizes the metric it's then graded on.
- **Why it fails (if it does):** The optimization window and the scoring window are identical; no strict prefix separation.
- **Revisit:** Pending **Issue #1336** @0hq ruling. Causal-context-only variant (mask=`[0:s]`, optimizing only strictly-prior context) is the legal candidate.
- **Ref:** Our 771-PR sweep flagged PRs #1321, #1324, #1278, #1263, #1319, #1376 in this cluster. One ruling clears or closes all.

---

## How to use this document

1. **Before you burn GPU:** check if your idea is here. If yes, read why it failed and decide whether your variant avoids the cause.
2. **If you retest and we're wrong:** open a PR to this document with your data. Negative results are as useful as positive ones.
3. **If you have your own dead-ends:** submit them as a PR. Community wins when failure data is pooled.

## What's NOT in this document (on purpose)

- Techniques that only fail at scales we haven't tested (larger vocabs, different architectures)
- Private-run observations without documented deltas
- Anything where we're uncertain whether the failure was the technique or our implementation

---

## References

- [PROTEUS V2 Plan](../../parameter-golf-private/PROTEUS_V2_PLAN.md) — original dead-end list
- [GDN decision memo](../../parameter-golf-private/research/gdn-should-we-bet-memo.md)
- [openai/parameter-golf Issue #677](https://github.com/openai/parameter-golf/issues/677) — compliance megathread
- [openai/parameter-golf Issue #1336](https://github.com/openai/parameter-golf/issues/1336) — SLOT ruling
- [openai/parameter-golf PR #779](https://github.com/openai/parameter-golf/pull/779) — n-gram family-bug ruling
- [openai/parameter-golf PR #1314](https://github.com/openai/parameter-golf/pull/1314) — Scylla corrected vocab
- Ciprian-Florin Ifrim PR #1388 (XNOR dossier) — set the community standard for negative-result transparency

## Attribution

Compiled by [@MatoTeziTanka](https://github.com/MatoTeziTanka) from PROTEUS (v6-v16) experimentation logs and the Agora's 771-PR compliance review. External confirmation for engram noted per Ciprian-Florin Ifrim's 2026-04-14 Discord post.

*AI-assisted compilation: data extraction and table formatting by Claude (Opus 4.6); every entry verified against source logs or PR references before publication.*
