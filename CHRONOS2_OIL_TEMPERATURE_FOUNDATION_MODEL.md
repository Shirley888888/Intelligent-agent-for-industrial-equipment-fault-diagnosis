# Transformer Oil-Temperature Time-Series Foundation Model (Chronos-2)

## 1. What this project is training

This stage does **not** train a new LSTM/TCN and does not claim that the current ETTh1 predictor is already a foundation model. It starts from the public 120M-parameter Chronos-2 time-series foundation model (`amazon/chronos-2`) and fine-tunes it for the transformer oil-temperature forecasting task.

- Target: `OT` (oil temperature, degrees Celsius).
- Historical covariates: `HUFL,HULL,MUFL,MULL,LUFL,LULL`.
- Future load is **not** assumed to be known. All six load variables are supplied as past-only covariates.
- Hourly and 15-minute data are trained separately.
- LSTM and TCN remain local supervised baselines; they are not described as foundation models.

Official model repository: https://github.com/amazon-science/chronos-forecasting
Model card: https://huggingface.co/amazon/chronos-2
Fine-tuning implementation follows `Chronos2Pipeline.fit()` and its documented `inputs`, `validation_inputs`, `finetune_mode='lora'`, `prediction_length`, `context_length`, `learning_rate` and `num_steps` API.

## 2. Real transformer data source

Official data repository: https://github.com/zhouhaoyi/ETDataset

ETT-small describes about two years of oil-temperature and six load variables collected from a real operating platform associated with Beijing Guowang Fuda Science & Technology Development Company. It contains two station/region series and two sample frequencies. Upstream descriptions give a theoretical 70,080 points for each 15-minute series, while a secondary parsed dataset card reports 69,680; the downloader deliberately records the actual downloaded CSV row count instead of assuming either value:

| File | Interval | Rows (upstream description) | Forecast horizon with `prediction_length=24` |
|---|---:|---:|---:|
| ETTh1 | 1 hour | 17,420 | 24 hours |
| ETTh2 | 1 hour | 17,420 | 24 hours |
| ETTm1 | 15 minutes | about 69.7k (verify actual CSV) | 6 hours |
| ETTm2 | 15 minutes | about 69.7k (verify actual CSV) | 6 hours |

Important: these are **two physical station/region series**, each published at two sampling frequencies. ETTh1 and ETTm1 are not separate sites; they are the hourly and 15-minute series for region 1. ETTh2 and ETTm2 correspond to region 2. Do not count the four CSV files as four independent transformers. Only ETT-small is marked public upstream. ETT-large (39 transformers) and ETT-full (69 transformers) are still marked unavailable; this project does not claim to have downloaded them.

Run the byte-preserving source downloader:

```bash
python download_ett.py --output-dir data/ett
```

It downloads each original CSV byte-for-byte from a pinned upstream commit (`1d16c8f4f943005d613b5bc962e9eeb06058cf07`). In addition to expected columns, finite numeric values, row count, and timestamp cadence, it checks each file against its Git blob SHA-1 from the official GitHub Contents API, plus a local SHA-256. The file-level hashes are pinned in `download_ett.py`; provenance is stored in `data/ett/source_manifest.json`. The package currently contains the already-provided ETTh1 source file only; the other three official files are fetched by the downloader on the networked workstation.

## 3. Environment

Use a separate Python 3.10+ environment for Chronos-2. This avoids unexpectedly replacing the current PyTorch / Transformers stack used by the existing online-learning experiments.

```bash
conda create -n oil_chronos python=3.11 -y
conda activate oil_chronos
# Install a CUDA-enabled PyTorch build matching the workstation's CUDA driver first.
pip install -r requirements_chronos.txt
```

The initial run requires Internet access to download `amazon/chronos-2` model weights from Hugging Face. `--device-map cuda` is recommended on a supported GPU; `--device-map auto` lets Hugging Face choose a supported device. CPU fine-tuning is possible but can be substantially slower.

## 4. Fine-tune the foundation model

Hourly group (two station series, 96 input points -> 24 future hours):

```bash
python train_chronos2_ett.py --data-dir data/ett --frequency hourly \
  --out-dir models/chronos2_ett --device-map cuda --finetune-mode lora \
  --context-length 512 --prediction-length 24 --num-steps 1000 --batch-size 4 --gradient-accumulation-steps 4
```

15-minute group (two station series, 96 points = 24 hours of history -> 24 points = 6 future hours):

```bash
python train_chronos2_ett.py --data-dir data/ett --frequency 15min \
  --out-dir models/chronos2_ett --device-map cuda --finetune-mode lora \
  --context-length 512 --prediction-length 24 --num-steps 1000 --batch-size 4 --gradient-accumulation-steps 4
```

To use full fine-tuning instead of LoRA, change `--finetune-mode full`. Do not combine the two sampling frequencies in one training run.

## 5. Evaluation and candidate gate

For each frequency and each station, the time axis is divided chronologically into **60% / 10% / 10% / 20%**:

- **0–60% training prefix**: the only portion used to build random training windows;
- **60–70% fine-tuning validation prefix**: supplied to `Chronos2Pipeline.fit()` for its internal validation/early-stopping callback; its final prediction horizon is the validation target;
- **70–80% reserved candidate-gate segment**: never passed to `fit()` or its early-stopping callback. The zero-shot base and fine-tuned candidate are compared on identical rolling-origin windows here;
- **80–100% final test segment**: never passed to `fit()` and never used by candidate-gate decisions. It is evaluated once for independent reporting and human review.

The rolling evaluations use the same origins and forecast horizon for the zero-shot and fine-tuned models. Metrics include aggregate and per-station MAE, RMSE, P95, P99, MaxAE, P10-P90 interval coverage and mean interval width.

### Cross-station transfer evaluation (recommended before pooled fine-tuning)

ETT-small has only two station/region series per frequency, so the package also includes four leave-one-station-out (LOSO) run configurations:

| Frequency | Fine-tune on | Gate/test on | Config |
|---|---|---|---|
| Hourly | ETTh1 | ETTh2 | `configs/chronos2_ett_hourly_loso_h1_to_h2.json` |
| Hourly | ETTh2 | ETTh1 | `configs/chronos2_ett_hourly_loso_h2_to_h1.json` |
| 15-minute | ETTm1 | ETTm2 | `configs/chronos2_ett_15min_loso_m1_to_m2.json` |
| 15-minute | ETTm2 | ETTm1 | `configs/chronos2_ett_15min_loso_m2_to_m1.json` |

Example:

```bash
python train_chronos2_ett.py --config configs/chronos2_ett_hourly_loso_h1_to_h2.json --device-map cuda
python train_chronos2_ett.py --config configs/chronos2_ett_hourly_loso_h2_to_h1.json --device-map cuda
```

In LOSO, the evaluation station is excluded from gradient updates and fine-tuning validation. Its historical observations up to each rolling forecast origin are still used as model context, as they would be available when forecasting oil temperature at a newly onboarded transformer. This is **cross-station transfer with target-station historical context**, not a no-history cold-start experiment. The final 20% of the evaluation station remains untouched until final reporting.

After the two directional transfer runs, the pooled two-station configurations (`chronos2_ett_hourly.json` and `chronos2_ett_15min.json`) can be used to create the final frequency-specific domain-adapted candidates. This order exposes negative transfer instead of hiding it in a pooled average.

The independent candidate gate is run on the reserved 70–80% segment. Initial thresholds: aggregate MAE regression <= 1%; aggregate P99 regression <= 5%; aggregate MaxAE regression <= 10%; each station MAE regression <= 3%; P10-P90 empirical coverage >= 70% overall and per station; mean interval width regression <= 20%. The interval has nominal 80% coverage, so 70% is an initial under-coverage red line, not a calibration guarantee. These thresholds require review against operating risk and more site data. The final 20% test metrics do not participate in candidate selection. Even after the gate passes, the checkpoint is only a candidate: independent test review, the project model gate, shadow replay and the joint release registry must still pass. No automatic activation occurs.

Artifacts are saved under `models/chronos2_ett/<frequency>/`, including:

- `finetuned-ckpt/`: Hugging Face Chronos-2 fine-tuned checkpoint;
- `manifest.json`: model/base-model revisions, data hashes, split rules, horizon, metrics, gate state and checkpoint-tree SHA-256;
- `zero_shot_metrics.json`, `candidate_metrics.json`: final untouched test-segment metrics;
- `gate_validation_zero_shot_metrics.json`, `gate_validation_candidate_metrics.json`: reserved candidate-gate metrics;
- `candidate_gate.json`: independent 70–80% gate result;
- `zero_shot_predictions.jsonl`, `candidate_predictions.jsonl`: final test predictions;
- `gate_validation_predictions.jsonl`: reserved gate-segment predictions;
- `manifest.json`: includes all split boundaries, data hashes and both gate/test metrics.

## 6. Run the fine-tuned model through the shared Agent

```bash
python -m diagnostic_agent.cli run-chronos2-oil \
  --model-dir models/chronos2_ett/hourly \
  --csv data/ett/ETTh2.csv \
  --equipment-id ETT-STATION-2 \
  --workspace agent_workspace_chronos2 \
  --device-map cuda
```

The Chronos bridge passes OT point forecasts and P10/P50/P90 predictive quantiles into the same DiagnosticAgent and EpisodeStore used by the earlier LSTM/TCN flows. Predictive quantiles are **not** a calibrated fault probability, an OOD score, or a fault class. The Agent therefore keeps `confidence_calibrated=false`, leaves `fault_type=null`, and uses the existing conservative/manual-review policy unless separate validated risk detectors are provided.

## 7. Windows workstation quick start

From the project root in Anaconda Prompt, use a clean Python 3.11 environment. Install a PyTorch build compatible with the workstation's NVIDIA driver/CUDA first, then install the optional Chronos dependencies:

```bat
conda create -n oil_chronos python=3.11 -y
conda activate oil_chronos
python -m pip install --upgrade pip
REM Install the correct CUDA-enabled PyTorch build for this workstation before the next line.
python -m pip install -r requirements_chronos.txt
python download_ett.py --output-dir data/ett
```

Then fine-tune the two frequencies in separate runs (use a fresh `--out-dir` when repeating a run because the script refuses to overwrite an existing run):

```bat
python train_chronos2_ett.py --config configs/chronos2_ett_hourly.json --device-map cuda
python train_chronos2_ett.py --config configs/chronos2_ett_15min.json --device-map cuda
```

The first run requires GitHub access for the remaining original ETT-small CSVs and Hugging Face access to download `amazon/chronos-2`. Keep the original data and model caches on the training workstation. The provided configs use micro-batch 4 and gradient accumulation 4 (approximate effective batch 16) as a conservative starting point for an 8GB-class GPU. Reduce micro-batch to 2 or 1 if CUDA reports out-of-memory. Start with the supplied `--preflight-only` validation and then run `run_chronos2_oil_smoke_windows.bat` (20 steps per frequency, micro-batch 2) in fresh output directories before launching the full 1000-step run with `run_chronos2_oil_foundation_training_windows.bat`.

## 8. What is and is not claimed in this package

- The original ETTh1 file is bundled from the user's previous project input and its SHA-256 plus official Git blob SHA-1 are recorded. The four ETT-small files represent two station/region series at two different sampling frequencies, not four independent stations.
- The downloader and training pipeline support the four original ETT-small files. The upstream repository currently marks ETT-small as public, while ETT-large and ETT-full remain unpublished.
- ETTm1/ETTm2/ETTh2 remain to be downloaded on a machine with GitHub access; they are not bundled as transformed derivative copies.
- The Chronos-2 integration and mocked bridge tests are included; the local ETT source/Agent test suite has 5 tests and Chronos fine-tuning orchestration suite has 11 tests.
- Chronos-2 weights and a Chronos-2 fine-tuned checkpoint were **not** downloaded or trained in this build environment because it has no outbound DNS/network access and no CUDA GPU. Therefore this package makes no claim of a completed Chronos-2 fine-tuning run or Chronos-vs-LSTM/TCN score. Execute the fine-tuning commands above on the networked GPU workstation to produce that measured result. The package includes both same-station temporal evaluation and optional cross-station LOSO configurations.
- ETT-small has two series only. It is a real transformer-operating dataset and a useful domain fine-tuning benchmark, but it is not a fleet-scale multi-transformer pretraining corpus. Do not claim generalization to arbitrary transformer models, voltage classes or manufacturers without additional data.

## 9. Citations

- Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*, AAAI 2021; ETT dataset: https://github.com/zhouhaoyi/ETDataset
- Chronos-2 model and model card: https://huggingface.co/amazon/chronos-2
- Chronos-2 training and inference code: https://github.com/amazon-science/chronos-forecasting

## LoRA 依赖与版本保护

当前依赖下限为 `chronos-forecasting==2.3.2`。训练入口在调用 Chronos-2 `fit()` 前显式检查 `peft` 是否可导入；当配置要求 LoRA 而 PEFT 不存在时会立即报错，不允许 Chronos-2 静默回退到全量微调后仍把运行记录标成 LoRA。请用 `pip show chronos-forecasting peft transformers` 保存实际版本信息。
