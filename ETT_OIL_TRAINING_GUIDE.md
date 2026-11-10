# 真实变压器油温数据接入与模型训练指南

## 1. 数据边界

目标变量是 `OT`（Oil Temperature，变压器油温），不是轴承振动、故障类别或 RUL。特征列固定为 `HUFL,HULL,MUFL,MULL,LUFL,LULL,OT`。

官方 ETT-small 提供两个站点，小时级 `ETTh1/ETTh2` 和 15 分钟级 `ETTm1/ETTm2`。本项目将两种频率分开建模：

- hourly：两站点共享 LSTM 或 TCN，96 小时历史预测未来 24 小时。
- 15min：两站点共享 LSTM 或 TCN，96 个 15 分钟点（24 小时）预测未来 24 个点（6 小时）。

这是真实变压器数据上的多站点、多频率预测实验，不应称为覆盖数百台设备的通用“油温大模型”。

## 2. 下载原始数据

在有 GitHub 网络连接的训练机上，从项目根目录运行：

```bash
python download_ett.py --output-dir data/ett
```

将下载并逐个校验 ETTh1、ETTh2、ETTm1、ETTm2。校验包括列名、有限值、时间排序、重复时间/采样间隔以及 SHA-256，来源信息记录在 `data/ett/source_manifest.json`。

如果只是先运行小时组：

```bash
python download_ett.py --output-dir data/ett --names ETTh1 ETTh2
```

## 3. 训练

训练小时级 LSTM 与 TCN：

```bash
python train_ett_corpus.py --data-dir data/ett --frequency hourly --model both --out-dir models/ett_corpus --device cuda:0 --epochs 10 --patience 3
```

训练 15 分钟级 LSTM 与 TCN：

```bash
python train_ett_corpus.py --data-dir data/ett --frequency 15min --model both --out-dir models/ett_corpus --device cuda:0 --epochs 10 --patience 3
```

一次运行两种频率（总共训练四个模型）：

```bash
python train_ett_corpus.py --data-dir data/ett --frequency all --model both --out-dir models/ett_corpus --device cuda:0 --epochs 10 --patience 3
```

如果只下载了部分数据，可以用 `--datasets ETTh1` 做单站点 smoke/基线，但这不能作为双站点泛化结果。

## 4. 数据隔离和指标

每条变压器序列独立按时间顺序划分 60%/20%/20%。窗口不会跨设备边界；归一化器只使用各站点训练前缀拟合；验证集只用于早停和选 epoch；测试集只在训练结束后评估一次。

输出每个设备以及汇总的 MAE、RMSE、P95 绝对误差、P99 绝对误差和最大绝对误差，保存在每个模型目录的 `metrics.json`。训练历史、数据哈希、分割边界和 Checkpoint 哈希写入 `manifest.json`。

注意：两个站点的测试集是时间外推，不是完全未见站点测试，因为两个站点都参与训练。跨站点泛化应另做 leave-one-station-out 实验。

## 5. 接入同一个 Agent

训练完成后，使用生成的模型目录运行 Agent。例如：

```bash
python -m diagnostic_agent.cli run-ett-corpus \
  --model-dir models/ett_corpus/hourly/LSTM \
  --csv data/ett/ETTh2.csv \
  --equipment-id ETT-STATION-2 \
  --workspace agent_workspace_oil_temp \
  --device cuda:0
```

15 分钟模型将 `--model-dir` 指向 `models/ett_corpus/15min/LSTM`，CSV 用 `ETTm1.csv` 或 `ETTm2.csv`。Agent Episode 会记录模型 ID、CheckPoint SHA-256、数据来源、采样频率和 24 步预测。

当前 Agent 的风险控制仍会把没有独立校准置信度/OOD 证据的纯回归预测交由人工复核；不要将温度回归数值改名为故障分类置信度。


## 6. 时序基础大模型（Chronos-2）

如果目标是油温预测时间序列基础模型，而不是仅训练 LSTM/TCN，请先阅读 `CHRONOS2_OIL_TEMPERATURE_FOUNDATION_MODEL.md`。该流程从 `amazon/chronos-2` 预训练基础模型开始，使用 ETT-small 的 OT 及六个历史负载协变量进行 LoRA/全量微调。

```bash
# 安装可选依赖（建议独立 Python 3.11 环境）
pip install -r requirements_chronos.txt

# 下载四个官方 ETT-small 数据文件
python download_ett.py --output-dir data/ett

# 小时级和 15 分钟级分别微调 Chronos-2
python train_chronos2_ett.py --frequency hourly --out-dir models/chronos2_ett --device-map cuda --finetune-mode lora
python train_chronos2_ett.py --frequency 15min --out-dir models/chronos2_ett --device-map cuda --finetune-mode lora

# 让微调后的 Chronos-2 进入共享 Agent
python -m diagnostic_agent.cli run-chronos2-oil --model-dir models/chronos2_ett/hourly --csv data/ett/ETTh2.csv --equipment-id ETT-STATION-2 --device-map cuda
```

本流程采用 60% 训练、10% Chronos 内部验证、10% 独立候选门禁、20% 最终测试；门禁与最终测试不复用同一段未来数据。它会评估 MAE、P95/P99、MaxAE 及 P10-P90 覆盖率与区间宽度。通过 70%-80% 候选门禁不代表自动上线，最终 20% 仍须独立审阅，且要经过项目 Model Gate 与 Agent 联合发布。

**当前执行环境的实际限制：** 本次构建容器没有外网 DNS、没有 `chronos-forecasting` / Hugging Face 模型缓存，也没有 CUDA；因此未在这里下载 Chronos-2 权重或完成真实 LoRA 微调。已完成的是官方接口兼容的训练脚本、原始数据下载器、数据划分、门禁、Agent 桥接以及模拟推理测试。不要把原有 ETTh1 LSTM/TCN 训练结果说成 Chronos-2 大模型训练结果。请在可访问 GitHub/Hugging Face 且具备合适 GPU 的训练机上执行上述命令，产生实际的 Chronos checkpoint 和实测指标。


## Chronos-2 跨站点评估补充

ETT-small 实际只有两个站点/区域序列；`h1/m1` 是区域 1 的小时/15 分钟版本，`h2/m2` 是区域 2 的小时/15 分钟版本，并非四台独立设备。正式做 pooled fine-tuning 之前，建议先执行双向 LOSO：

```bash
python train_chronos2_ett.py --config configs/chronos2_ett_hourly_loso_h1_to_h2.json --device-map cuda
python train_chronos2_ett.py --config configs/chronos2_ett_hourly_loso_h2_to_h1.json --device-map cuda
python train_chronos2_ett.py --config configs/chronos2_ett_15min_loso_m1_to_m2.json --device-map cuda
python train_chronos2_ett.py --config configs/chronos2_ett_15min_loso_m2_to_m1.json --device-map cuda
```

LOSO 的目标站点不参与参数训练与微调验证，但它在每个滚动预测起点之前的历史数据可以作为预测上下文；因此这是跨站点迁移评估，不是目标站点完全无历史的冷启动。完成双向评估后，再运行双站点 pooled 配置生成小时级与 15 分钟级候选。

## LoRA 依赖与版本保护

当前依赖下限为 `chronos-forecasting==2.3.2`。训练入口在调用 Chronos-2 `fit()` 前显式检查 `peft` 是否可导入；当配置要求 LoRA 而 PEFT 不存在时会立即报错，不允许 Chronos-2 静默回退到全量微调后仍把运行记录标成 LoRA。请用 `pip show chronos-forecasting peft transformers` 保存实际版本信息。

## Chronos-2 GPU smoke / full training scripts

- `run_chronos2_oil_smoke_windows.bat`: downloads and SHA-verifies ETT-small, checks hourly/15-minute preflight, then runs 20-step LoRA smoke jobs.
- `run_chronos2_oil_foundation_training_windows.bat`: after smoke review, runs the full 1000-step hourly and 15-minute LoRA fine-tuning jobs.
- Default per-device micro-batch 4 + gradient accumulation 4; smoke uses micro-batch 2 + accumulation 2. Reduce micro-batch if CUDA OOM occurs.

See `CHRONOS2_EXECUTION_STATUS.md` for the exact environment limits and the distinction between mocked integration tests and real model training.
