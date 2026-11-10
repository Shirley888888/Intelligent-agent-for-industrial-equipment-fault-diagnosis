# Chronos-2 油温微调 GPU Smoke：实测 Checkpoint 与门禁报告

生成时间：2026-11-02。本报告记录在本机 NVIDIA GPU 上**真实执行**的 Chronos-2 LoRA 微调与候选门禁评估，不是 mock 测试，也不替代正式 1000-step 训练。

## 1. 结论摘要

| 频率 | 真实 Checkpoint | 门禁结论 | 说明 |
|---|---|---|---|
| hourly (ETTh1/ETTh2, 96→24 小时) | `models/chronos2_smoke_hourly/hourly/finetuned-ckpt/` | **通过（7/7 项）** | 非回归门禁通过；20 步 LoRA 改动极小，MAE 相对 zero-shot +0.0057%，**不是精度提升结论** |
| 15min (ETTm1/ETTm2, 96→24 个 15 分钟点) | `models/chronos2_smoke_15min/15min/finetuned-ckpt/` | **拒绝（6/7 项）** | 唯一失败项：`ETTm2` 的 P10–P90 覆盖率 0.6731 < 门禁下限 0.70 |

两份产物均为**候选**，`manifest.json` 中 `agent_activation=false`；hourly 的 `release_status` 为 `gate_validation_passed_pending_independent_test_and_joint_release_review`。门禁通过**不会**自动发布模型。

Checkpoint 性质：`finetuned-ckpt/` 是 **PEFT LoRA 适配器**（`adapter_config.json` + `adapter_model.safetensors`，约 4.63 MiB，r=8、alpha=16、target 为 5 类注意力/输出投影模块），**不是**独立的 120M 全量权重；加载时仍需 `amazon/chronos-2` 基座权重（commit `29ec3766d36d6f73f0696f85560a422f50e8498c`）。

## 2. 执行环境（实测）

| 项 | 值 |
|---|---|
| OS / Python | Windows 11 10.0.26200 / Python 3.12.7 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8150 MiB，SM 12.0 |
| PyTorch | `2.12.0.dev20260408+cu128`，CUDA 12.8，`torch.cuda.is_available()=True` |
| chronos-forecasting | 2.3.2 |
| peft / accelerate / transformers | 0.21.2 / 1.15.0 / 5.15.0 |
| numpy | 1.26.4 |

`pip install --dry-run -r requirements_chronos.txt` 显示仅新增 `einops/accelerate/peft/chronos-forecasting`，未替换现有 CUDA torch 或 transformers；安装后 `tests_chronos` 11 项、`tests_ett` 5 项仍全部通过。

## 3. 数据来源核验

上游固定提交 `1d16c8f4f943005d613b5bc962e9eeb06058cf07`，四份 CSV 均通过列名、时间戳间隔、数值有限性、行数及 **Git blob SHA-1** 校验（记录见 `data/ett/source_manifest.json`）。

| 文件 | 行数 | 频率 | SHA-256（前 16 位） | Git blob SHA-1（前 16 位） |
|---|---:|---|---|---|
| ETTh1.csv | 17420 | 1h | `f18de3ad269cef59` | `a52c4925778c07c1` |
| ETTh2.csv | 17420 | 1h | `a3dc2c597b9218c` | `6e3760778598fdee` |
| ETTm1.csv | 69680 | 15min | `6ce1759b1a18e33` | `62c9f979e2dc5b0` |
| ETTm2.csv | 69680 | 15min | `db973ca252c6410a` | `f9d46027c987d4` |

两个频率的 `--preflight-only` 均通过，记录见 `verification_chronos2/preflight_hourly_gpu.json` 与 `preflight_15min_gpu.json`。时间切分为 60/10/10/20：hourly 训练 `[0,10452)`、HF 验证 `[0,12194)`、门禁 `[12194,13936)`、最终测试 `[13936,17420)`；15min 分别为 `[0,41808)`、`[0,48776)`、`[48776,55744)`、`[55744,69680)`。门禁区间不进入 `fit()`，最终 20% 不参与候选选择。

## 4. 执行命令

```bat
python -m pip install -r requirements_chronos.txt
python download_ett.py --output-dir data/ett
python train_chronos2_ett.py --config configs/chronos2_ett_hourly.json --preflight-only
python train_chronos2_ett.py --config configs/chronos2_ett_15min.json --preflight-only
python train_chronos2_ett.py --config configs/chronos2_ett_hourly.json --device-map cuda --num-steps 20 --batch-size 2 --gradient-accumulation-steps 2 --out-dir models/chronos2_smoke_hourly
python train_chronos2_ett.py --config configs/chronos2_ett_15min.json --device-map cuda --num-steps 20 --batch-size 2 --gradient-accumulation-steps 2 --out-dir models/chronos2_smoke_15min
```

smoke 配置：`finetune_mode=lora`、`num_steps=20`、`batch_size=2`、`gradient_accumulation_steps=2`（有效 batch 4）、`context_length=512`、`prediction_length=24`、`lr=1e-5`、`seed=42`、`test_stride=24`。hourly 全进程约 98 秒，15min 约 8 分钟。日志：`verification_chronos2/smoke_hourly_gpu.log`、`smoke_15min_gpu.log`（PowerShell `>` 重定向写出为 UTF-16LE BOM，读取时需按此编码）。

## 5. 真实训练证据

`checkpoint-20/trainer_state.json`（hourly）：`max_steps=20`、`train_batch_size=2`、step 20 `loss=0.602674`、`grad_norm=1.0437`、`learning_rate=5e-7`、`eval_loss=0.275564`、`train_runtime=11.55s`、`eval_samples_per_second=80.53`。15min 同样是 20 步真实训练。零样本基座与微调候选分别在预留门禁区间和最终测试区间各做一次滚动预测，共 4 组评估；`zero_shot_predictions.jsonl`、`candidate_predictions.jsonl`、`gate_validation_predictions.jsonl` 保存逐窗口预测、P10/P90 与真值。

## 6. 实测门禁报告

门禁阈值（首轮工程阈值）：整体 MAE 回归 ≤ 1%、各站 MAE 回归 ≤ 3%、P99 回归 ≤ 5%、MaxAE 回归 ≤ 10%、P10–P90 覆盖率 ≥ 0.70、区间宽度回归 ≤ 20%。

### 6.1 hourly（144 个门禁窗口 / 290 个最终测试窗口）

| 指标 | zero-shot | 微调候选 | 门禁判定 |
|---|---:|---:|---|
| 门禁 MAE (℃) | 1.561141 | 1.561230 | 整体回归 +0.0057% ≤ 1%，通过 |
| 门禁 P99 AE (℃) | 8.6363 | 8.6389 | +0.0303% ≤ 5%，通过 |
| 门禁 MaxAE (℃) | 12.9033 | 12.7947 | −0.8410%，通过 |
| 门禁 P10–P90 覆盖率 | 0.7558 | 0.7569 | ≥ 0.70，通过 |
| 门禁平均区间宽度 (℃) | 4.3922 | 4.3985 | +0.1420% ≤ 20%，通过 |
| ETTh1 / ETTh2 MAE 回归 | — | — | +0.0240% / −0.0038%，均 ≤ 3%，通过 |
| ETTh1 / ETTh2 覆盖率 | — | 0.7824 / 0.7315 | 均 ≥ 0.70，通过 |

**`candidate_gate.json`: `passed=true`，7 项全部通过。**

未参与门禁的最终 20% 测试集（仅作独立审查）：MAE 1.901170 → 1.900869 ℃，RMSE 3.017997 → 3.017630 ℃，P99 12.3433 → 12.3905 ℃，覆盖率 0.7603 → 0.7615。差异在 20 步 LoRA 下属于微小波动，不能作为性能收益。

### 6.2 15min（580 个门禁窗口 / 1160 个最终测试窗口）

| 指标 | zero-shot | 微调候选 | 门禁判定 |
|---|---:|---:|---|
| 门禁 MAE (℃) | 0.933101 | 0.932197 | −0.0969%，通过 |
| 门禁 P99 AE (℃) | 6.7472 | 6.7374 | −0.1443%，通过 |
| 门禁 MaxAE (℃) | 14.2385 | 14.1359 | −0.7204%，通过 |
| 整体 P10–P90 覆盖率 | 0.7195 | 0.7208 | ≥ 0.70，通过 |
| ETTm1 / ETTm2 MAE 回归 | — | — | +0.0347% / −0.1648%，通过 |
| **ETTm2 P10–P90 覆盖率** | — | **0.6731** | **< 0.70，失败** |

**`candidate_gate.json`: `passed=false`**，唯一失败项为 `each_station_interval_coverage_floor`。这是实测拒绝，不是脚本错误；`release_status=candidate_rejected_by_gate_validation`。最终测试集（独立审查）：MAE 0.898074 → 0.896840 ℃，覆盖率 0.7761 → 0.7780。

## 7. Checkpoint 完整性与可加载性

| 频率 | `model_artifact_sha256_tree`（实测重算，与 manifest 一致） |
|---|---|
| hourly | `5fd2e03d764f61b6bf62efb1096324fd1bfa171a7ea082a8e4ca8d2325a85d80` |
| 15min | `fe6bb2294fe0869e827869af355f3bc0e294fe4baf34b10b58fe10623825a991` |

已用项目自带 `Chronos2OilTemperatureBridge` 实际加载 hourly 产物：manifest 校验、SHA-256 树校验通过，`Chronos2Pipeline` 在 `cuda:0` 上加载成功，并对 `data/ett/ETTh2.csv` 生成了完整 Agent 输入上下文。另对比基座与微调 checkpoint 在同一输入上的 24 步中位数预测，最大差异 0.00224 ℃、`np.allclose=False`，确认 LoRA 适配器确实生效（差异极小符合 20 步 smoke 预期）。

## 8. 边界与不可声称事项

1. 这是 **20 步 smoke**，用于验证依赖、权重下载、CUDA 训练、评估、门禁与产物链路；不是正式 1000-step 微调，不得作为 Chronos-2 与 LSTM/TCN 的精度对比结论。
2. 产物是 LoRA 适配器，需基座权重；`finetuned-ckpt/` 不含 120M 全量参数。
3. 门禁是「相对 zero-shot 的非回归」检查，hourly 通过不代表有显著改善；15min 被拒是覆盖率下限触发，其他 6 项均通过。
4. 预测分位数不是故障置信度，不是 OOD/漂移检测信号；Agent 侧仍按 fail-closed 请求人工复核。
5. 两个候选均未激活（`agent_activation=false`），未经过独立测试集人工审查、项目模型门禁、影子回放与联合版本登记。
6. ETT-small 只有两个站点/区域在两种采样频率下的序列，不是多设备机队语料；不能据此声称对任意厂家/容量变压器的泛化能力。原始数据许可 CC BY-ND 4.0，本包不重新分发原始 CSV。
7. `CHRONOS2_EXECUTION_STATUS.md` 与 `CHRONOS2_OIL_TEMPERATURE_README_CN.md` 中「本次构建环境未执行真实训练」的描述针对原离线构建环境，现已过时，发布前需另行更新。

## 9. 产物清单

```
models/chronos2_smoke_hourly/hourly/
  finetuned-ckpt/{adapter_config.json, adapter_model.safetensors, README.md}
  checkpoint-20/{trainer_state.json, training_args.bin, adapter_*, README.md}
  manifest.json  candidate_gate.json
  zero_shot_metrics.json  candidate_metrics.json
  gate_validation_zero_shot_metrics.json  gate_validation_candidate_metrics.json
  zero_shot_predictions.jsonl  candidate_predictions.jsonl  gate_validation_predictions.jsonl
models/chronos2_smoke_15min/15min/   （同上结构）
verification_chronos2/
  preflight_hourly_gpu.json  preflight_15min_gpu.json
  smoke_hourly_gpu.log  smoke_15min_gpu.log  GPU_SMOKE_GATE_REPORT.md
data/ett/source_manifest.json
```
