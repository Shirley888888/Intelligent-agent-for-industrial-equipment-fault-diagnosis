# Chronos-2 Oil-Temperature Foundation Model Integration

## Included
- Added official `amazon/chronos-2` LoRA/full fine-tuning entry point for ETT-small OT forecasting.
- Added source-verifying downloader for official ETT-small ETTh1/ETTh2/ETTm1/ETTm2 CSVs.
- Added oil-temperature Agent bridge that records quantile forecasts, source-data SHA-256, model artifact SHA-256 and sampling frequency in the shared Episode.
- Added hourly and 15-minute configs and Windows workstation commands.
- Kept existing ETTh1 LSTM/TCN checkpoints and trainers as supervised baselines.

## Evaluation changes
- Split each station chronologically into 60% training, 10% HF fine-tune validation, 10% reserved candidate-gate validation, and 20% final test.
- The candidate gate is run only on the reserved 70%-80% segment, disjoint from HF early stopping and from final test.
- Gate checks overall MAE, P99 absolute error, MaxAE, per-station MAE, empirical P10-P90 coverage, and mean prediction-interval width.
- Final 20% metrics are written separately for human review. A candidate is never auto-activated by this script.

## Verification
- `tests_chronos`: 11 tests pass, including fake-pipeline fine-tune orchestration, gate/test window separation, tail-error gates, interval coverage/width gates, checkpoint hash checks and Agent Episode integration.
- The current build environment had no outbound network/DNS, no Chronos/Hugging Face model cache and no CUDA. Therefore actual Chronos-2 weight download and LoRA fine-tuning were not executed in this environment. No fine-tuned Chronos-2 checkpoint or performance result is claimed.

## 2026-11-02 — GitHub source pinning, preflight, and GPU-safe LoRA defaults

- Pinned ETT-small to upstream Git commit `1d16c8f4f943005d613b5bc962e9eeb06058cf07`.
- Recorded the expected Git blob SHA-1 for all four public files in the downloader and `source_manifest.json`; the downloaded files must match exactly before training.
- Added `--preflight-only` to validate dataset availability, checksums, sampling cadence, forecast horizon, chronological split boundaries and output directory without loading Chronos weights.
- Added `--run-name` for distinct leave-one-station-out output folders.
- Added gradient accumulation as an explicit Chronos-2 Trainer argument. Default micro-batch is 4 with accumulation 4 (approximate effective batch 16) to reduce peak GPU memory; reduce micro-batch further if needed.
- Added `run_chronos2_oil_foundation_training_windows.bat` for download/hash verification, preflight and separate hourly/15-minute LoRA runs.
- Updated optional requirements to `chronos-forecasting==2.3.2` and `peft>=0.20,<1`, matching the current Chronos 2.3.2 LoRA adapter compatibility notes.
- Added explicit execution status. The current build host has CPU-only PyTorch, no CUDA, no outbound DNS/network, and does not have Chronos/Transformers/PEFT/Accelerate installed; actual model weight download and fine-tuning are not claimed as completed.
- Validation: ETT source pipeline tests 5/5; Chronos integration tests 11/11; Agent 29/29; model-Agent integration 8/8; dual-evolution V0 12/12; dual-task integration 9/9. These tests verify local data integrity and mocked Chronos orchestration; they do not substitute for real Chronos training.
