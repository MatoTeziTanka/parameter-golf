# PROTEUS Combined — Parameter Golf Submission

**Built with [PROTEUS](https://lightspeedup.com) by LightSpeedUp**

## Approach

Four published techniques stacked on the baseline:

1. **EMA weight averaging** (Polyak 1992) — decay=0.999, fp32, every 10 steps. Smooths weight distributions for reduced INT8 quantization loss.
2. **seq_len=2048** — train and evaluate with longer context. Each token sees more information.
3. **FP16 embedding passthrough** — keep `tok_emb.weight` at FP16 instead of quantizing to INT8. The tied embedding pulls double duty as the output head — precision matters most here.
4. **Sliding window evaluation** (stride=64) — score each token with ~960 tokens of context instead of ~512. Pure eval-time improvement, zero training cost.

Plus hyperparameter tuning: `WARMDOWN_ITERS=3600`, `MATRIX_LR=0.06`, `SCALAR_LR=0.06`, `TIED_EMBED_LR=0.04`.

## Configuration

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key env vars (all have defaults baked into the script):
- `TRAIN_SEQ_LEN=2048`
- `WARMDOWN_ITERS=3600`
- `MATRIX_LR=0.06`
- `SCALAR_LR=0.06`
- `TIED_EMBED_LR=0.04`
- `EMA_ENABLED=1`
- `EMA_DECAY=0.999`
- `EMA_EVERY=10`
- `EVAL_STRIDE=64`
- `EVAL_BATCH_SEQS=512`

Architecture unchanged from baseline:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied embeddings: `TIE_EMBEDDINGS=1`
- Batching: `TRAIN_BATCH_TOKENS=524288`

## Key Metrics

- Timed training stopped at `12977/20000` steps due to wallclock cap
- Pre-quant eval at stop: `val_loss:2.0632`, `val_bpb:1.2219`
- **Post-quant sliding window eval: `val_loss:2.0074`, `val_bpb:1.1889`**
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.18888355`
- Train time: `600057ms` (`step_avg:46.24ms`)
- Eval time: `73032ms` (sliding window, stride=64)
- Peak memory: `10249 MiB allocated`

### Training volume

- Global batch: `524288` tokens/step
- Total train tokens seen: `~6.8B`

## Platform

Run on Modal 8×H100 SXM. Pending verification on RunPod 8×H100 SXM (official hardware).

## Included Files

- `train_gpt.py` — training script (baseline + EMA + sliding eval + FP16 embed)
- `train.log` — full training log from the submission run
- `submission.json` — leaderboard metadata
- `README.md` — this file
