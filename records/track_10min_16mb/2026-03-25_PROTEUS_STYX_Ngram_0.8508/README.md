# PROTEUS+STYX: LeakyReLU(0.9)² + 5-gram Eval Cache

**val_bpb:** 0.8508 (3-seed mean, std 0.0006)
**Improvement over SOTA (#549):** -0.269 BPB

## Architecture

PR #549 base stack with two modifications:

1. **LeakyReLU(0.9)²** — `F.leaky_relu(x, 0.9).square()` replacing the standard 0.5 slope. Based on our 7-point monotonic sweep (0.1–0.9) showing higher slope = lower BPB at this model scale.

2. **Backward-looking 5-gram eval cache** — numpy hash table (4M buckets) built from already-scored tokens during sliding window eval. Fixed-alpha blending: `p_final = 0.8 * p_model + 0.2 * p_cache`. No safety gate, no target-aware selection, no training data access.

| Parameter | Value |
|-----------|-------|
| Layers | 11 |
| Dimension | 512 |
| Heads | 8 (4 KV, GQA) |
| MLP | 3x (1536) |
| Activation | LeakyReLU(0.9)² |
| Vocab | 1024 BPE, tied embeddings |
| Quantization | INT6 + zstd |
| Cache | 5-gram, 4M buckets, alpha=0.2 |
| Eval stride | 64, seq_len=2048 |

## 3-Seed Results (8×H100 SXM)

| Seed | val_bpb | Cache Hit Rate | Eval Time |
|------|---------|----------------|-----------|
| 42   | 0.8513  | 98.2% | 155s |
| 1337 | 0.8502  | 98.2% | 134s |
| 2024 | 0.8510  | 98.2% | 134s |
| **Mean** | **0.8508** | **98.2%** | **std: 0.0006** |

## Verification: Not an Overlap Artifact

We verified the cache works at zero overlap (stride=2048):

| Stride | BPB | Hit Rate | Overlap |
|--------|-----|----------|---------|
| 64 (standard) | 0.8513 | 98.2% | 97% |
| 2048 (zero overlap) | 0.8709 | 97.9% | 0% |
| No cache | 1.1314 | — | — |

The 0.02 BPB gap between stride=64 and stride=2048 is the overlap contribution. The remaining 0.26 BPB improvement is genuine n-gram repetition in FineWeb.

## Compliance

- Cache is strictly backward-looking: tokens scored first, then added to cache
- No training data access during evaluation
- No oracle/hindsight selection (fixed alpha, always applied)
- No safety gate (no peeking at true token)
- Training ≤ 600s, evaluation ≤ 155s
- Artifact < 16MB
- Consistent with [Issue #677](https://github.com/openai/parameter-golf/issues/677) rules and the approach approved directionally by reviewers on [PR #674](https://github.com/openai/parameter-golf/pull/674)

## How the Cache Works

```python
ctx_table = np.zeros(4_194_304, dtype=np.uint32)
full_table = np.zeros(4_194_304, dtype=np.uint32)

# Per-token: look up 4-token context, blend if found
if ctx_table[ctx_hash] >= 2:
    p_ngram = min(full_table[full_hash], ctx_table[ctx_hash]) / ctx_table[ctx_hash]
    p_final = 0.8 * p_model + 0.2 * p_ngram

# After scoring window: update tables with scored tokens
```

## Logs

- `train_seed42.log`
- `train_seed1337.log`
- `train_seed2024.log`
- `verify_stride2048.log`

## Docker

`matotezitanka/proteus-pytorch:2.11.0-cuda12.8`

Built with [PROTEUS+STYX](https://lightspeedup.com) by Light Speed Up
