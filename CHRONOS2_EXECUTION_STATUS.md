# Chronos-2 / 变压器油温预测训练执行状态

## 当前已完成

1. 已确认任务为变压器油温 `OT` 多变量时间序列预测，不是故障分类、轴承诊断或 RUL 预测。
2. 已从官方 GitHub 仓库 `zhouhaoyi/ETDataset` 核验公开数据范围、字段定义和许可证。
3. 下载器固定到上游提交 `1d16c8f4f943005d613b5bc962e9eeb06058cf07`，并对 ETTh1/ETTh2/ETTm1/ETTm2 四个 CSV 校验 Git blob SHA-1、SHA-256、字段、采样间隔和行数。
4. 本包内已有的 ETTh1 原始文件通过官方 Git blob SHA-1 校验：`a52c4925778c07c1ef1a2cf6fd01594919717d9e`；本地 SHA-256 为 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`，17420 行，1 小时采样。
5. 已完成 Chronos-2 的 LoRA / 全量微调入口、按频率拆分的配置、same-station 时序门禁、cross-station LOSO 配置、预测区间指标、模型 Checkpoint 哈希和 Agent 桥接。
6. 已执行 ETTh1 单文件 preflight：确认 96 小时历史上下文、24 小时预测、60/10/10/20 时间切分及源文件哈希均通过。preflight 不加载模型权重、不开始训练。
7. `tests_ett` 5 项通过；`tests_chronos` 11 项通过。Chronos 测试使用 mock pipeline 验证数据处理、微调编排、门禁和 Agent 接口，不是实际模型权重训练。

## 当前没有完成的事项（必须明确）

- ETTh2、ETTm1、ETTm2 尚未下载到本次构建环境。
- `amazon/chronos-2` 预训练权重尚未下载。
- 本次构建环境为 Python 3.13.5、PyTorch 2.10.0+cpu，`torch.cuda.is_available()` 为 False；且未安装 `chronos`、`transformers`、`peft`、`accelerate`。
- 因此本次没有执行 Chronos-2 实际 LoRA 微调，没有产出 `finetuned-ckpt`，也没有 Chronos-2 对 LSTM/TCN 的真实精度比较。任何测试中的数值都属于 mock 测试，不得作为模型性能结果。

## 在有网络的 Windows + NVIDIA GPU 机器上执行

建议使用你此前的 Python 3.11 / CUDA PyTorch 环境。不要在没有备份时覆盖原环境的 torch；先确认 CUDA 可用，再安装可选 Chronos 依赖：

```bat
conda activate etth1_gpu
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -m pip install -r requirements_chronos.txt
```

先下载并校验全部官方 ETT-small CSV：

```bat
python download_ett.py --output-dir data/ett
```

先运行 preflight，不加载模型：

```bat
python train_chronos2_ett.py --config configs/chronos2_ett_hourly.json --preflight-only
python train_chronos2_ett.py --config configs/chronos2_ett_15min.json --preflight-only
```

再进行 Chronos-2 LoRA 域微调（建议先跑小步 smoke，再跑 1000 steps）：

```bat
python train_chronos2_ett.py --config configs/chronos2_ett_hourly.json --device-map cuda --num-steps 20 --batch-size 2 --gradient-accumulation-steps 2 --out-dir models/chronos2_smoke
python train_chronos2_ett.py --config configs/chronos2_ett_15min.json --device-map cuda --num-steps 20 --batch-size 2 --gradient-accumulation-steps 2 --out-dir models/chronos2_15m_smoke
```

确认显存和数据正常后，再执行完整配置：

```bat
python train_chronos2_ett.py --config configs/chronos2_ett_hourly.json --device-map cuda
python train_chronos2_ett.py --config configs/chronos2_ett_15min.json --device-map cuda
```

配置使用每设备 micro-batch 4、梯度累积 4（近似有效 batch 16）；若 GPU 显存不足，将 `batch_size` 改为 2 或 1，保持 gradient accumulation 或相应增加。任何重复运行请使用新的 `out_dir`，训练脚本不会覆盖已有候选。

## 训练结束后的验收顺序

1. 检查 `manifest.json` 中 `base_model_commit`、数据源 hash、时间区间、预测频率和预测跨度。
2. 检查 `candidate_gate.json`。如果 gate 未通过，候选不能发布。
3. 查看 zero-shot 与 fine-tuned 的同窗口指标：MAE、RMSE、P95、P99、MaxAE、按站点指标、P10-P90 coverage 和 interval width。
4. 单独审查未参与门禁的 final 20% 测试结果。
5. 运行 Agent 影子回放；预测区间不是校准故障置信度，不得让 Agent 从油温预测结果推断故障类别。
6. 经过联合版本登记和人工批准后，才将候选纳入生产发布流程。

## 数据范围与外推限制

ETT-small 是两个真实变压器站点/区域的运行数据；小时和 15 分钟数据是相同两个站点的不同采样频率，不是四个独立站点。ETT-large 与 ETT-full 在官方 README 中仍标为未公开。ETT-small 适合真实变压器油温预测的领域适配和方法验证，但不足以证明对任意厂家、容量、电压等级及全行业机队的泛化。
