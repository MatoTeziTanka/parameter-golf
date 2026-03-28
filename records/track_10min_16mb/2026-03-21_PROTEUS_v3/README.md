# PROTEUS v3 — Parameter Golf Submission

**Built with [PROTEUS](https://lightspeedup.com) by LightSpeedUp**

Continuation of [PR #95](https://github.com/openai/parameter-golf/pull/95) (PROTEUS v1). See that PR for our documented negative results on INT4 quantization, depth recurrence, and EMA overhead.

## Approach

Builds directly on thwu1's Int5-MLP submission and @unnir's SmearGate+OrthoInit (PR #162). We ran their stack on Modal 8×H100 due to persistent RunPod availability issues.

## Key Techniques (all from prior submissions)

- 10 layers, dim=512, MLP 3x (1536)
- Mixed int5/int6 quantization (int5 for MLP, int6 for attention)
- SmearGate + BigramHash(10240) + OrthoInit
- FP16 tied embedding + Late-K passthrough
- SWA (start_frac=0.4, every 50 steps)
- AdamW weight decay 0.04, grad clip 0.3
- Muon momentum 0.99, warmdown 3000
- Sliding window eval (stride=64, seq=2048)
- zstd-22 compression

## Key Metrics

- Training stopped at `6425/20000` steps due to wallclock cap
- Pre-quant eval: `val_loss:1.9620`, `val_bpb:1.1620`
- Post-quant sliding window eval: `val_loss:1.9425`, `val_bpb:1.1505`
- Exact: `final_int8_zlib_roundtrip_exact val_bpb:1.15045393`
- Train time: `599977ms` (`step_avg:93.38ms`)
- Eval time: `208172ms` (sliding window, stride=64)
- Artifact: `15,864,777 bytes`

## Platform

Run on Modal 8×H100. We experienced persistent RunPod provisioning issues (pods billing at GPU rate without reaching runtime) and have been unable to verify on RunPod SXM hardware. Seeking additional compute credits.

## Credits

- thwu1 — Int5-MLP + BigramHash(10240) + SWA architecture
- @unnir — SmearGate + BigramHash + OrthoInit (PR #162)
- @notapplica — Muon weight decay + spectral init (PR #60)
- @mattqlf — Sliding window eval (PR #50)

## Included Files

- `train_gpt.py` — training script
- `train.log` — full training log
- `submission.json` — leaderboard metadata
- `README.md` — this file
