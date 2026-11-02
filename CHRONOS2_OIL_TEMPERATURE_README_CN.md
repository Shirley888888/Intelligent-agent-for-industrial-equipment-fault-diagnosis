# 工业变压器油温预测基础模型：数据、微调与 Agent 接入指南

## 1. 本次目标

本项目不是从零训练一个通用大模型，而是使用公开预训练时间序列基础模型 **Amazon Chronos-2（120M）**，在真实变压器油温数据上做领域微调。现有 LSTM/TCN 保留为基线和现有在线学习实验，不替代 Chronos-2。

预测任务：用历史油温 `OT` 和六个历史负载变量 `HUFL/HULL/MUFL/MULL/LUFL/LULL`，预测未来油温。未来负载没有已知计划值，因此只提供历史协变量，不把未来负载泄漏给模型。

## 2. GitHub 真实数据源

官方仓库：<https://github.com/zhouhaoyi/ETDataset>

- ETTh1、ETTh2：小时级油温与六路负载；96→24 表示历史 96 小时预测未来 24 小时。
- ETTm1、ETTm2：15 分钟级油温与六路负载；96→24 表示历史 24 小时预测未来 6 小时。
- 这四个文件仅代表 **两个站点/区域在两种采样频率下的序列**，不是四台独立变压器。
- 官方仓库将 ETT-small 标为公开；ETT-large（39 台）和 ETT-full（69 台）仍未公开。
- 当前包只带原项目中的 ETTh1.csv；其余三份原始 CSV 在联网机器上运行下载器获取。

下载和检查：

```bash
python download_ett.py --output-dir data/ett
```

脚本检查列名、时间戳间隔、数值有限性和行数下限，并把每份文件的真实行数、起止时间和 SHA-256 写入 `data/ett/source_manifest.json`。它保留原始 CSV 字节，不做数据改写。

**许可提醒：** 官方 GitHub 仓库的 LICENSE 标示 CC BY-ND 4.0。保留来源与署名；项目不打包重新处理后的 ETTm1/ETTm2/ETTh2 文件。公开发布训练权重或商业使用前，应自行核对原始数据许可及模型权重的衍生材料问题。

## 3. 基础模型与训练方式

主候选：<https://huggingface.co/amazon/chronos-2>（官方代码：<https://github.com/amazon-science/chronos-forecasting>）。这是通用预训练时间序列基础模型，项目通过 LoRA 或全量微调形成“变压器油温领域候选模型”，不是在 ETT-small 上从头预训练一个 120M 大模型。

- 目标：`OT`，单位 ℃。
- 历史协变量：六路负载。
- 预测区间：P10/P50/P90。
- 小时级和 15 分钟级分开训练。
- LSTM/TCN 在同一数据切分上保留为基线，后续比较不能混用不同切分。

## 4. 推荐训练顺序

### A. 建立独立环境

建议使用 Python 3.11。先根据工作站 CUDA 驱动安装匹配的 CUDA 版 PyTorch，再安装 Chronos 相关依赖：

```bash
conda create -n oil_chronos python=3.11 -y
conda activate oil_chronos
python -m pip install --upgrade pip
# 先安装与你的 NVIDIA 驱动匹配的 CUDA 版 PyTorch
pip install -r requirements_chronos.txt
python download_ett.py --output-dir data/ett
```

首次运行需要能够访问 GitHub 下载三份数据，并访问 Hugging Face 下载 `amazon/chronos-2` 权重。

### B. 先做双向跨站点迁移测试

ETT-small 只有两个站点，因此先用一个站点微调、另一个站点评估，再反过来。目标站点不参与梯度更新和微调验证；但每个滚动预测点之前的目标站点历史 OT/负载仍作为预测上下文。这是“跨站点迁移 + 目标站点历史上下文”，不是完全无历史冷启动。

小时级：

```bash
python train_chronos2_ett.py --config configs/chronos2_ett_hourly_loso_h1_to_h2.json --device-map cuda
python train_chronos2_ett.py --config configs/chronos2_ett_hourly_loso_h2_to_h1.json --device-map cuda
```

15 分钟级：

```bash
python train_chronos2_ett.py --config configs/chronos2_ett_15min_loso_m1_to_m2.json --device-map cuda
python train_chronos2_ett.py --config configs/chronos2_ett_15min_loso_m2_to_m1.json --device-map cuda
```

### C. 双站点领域微调

跨站点表现可接受后，再用两个站点共同微调各频率模型：

```bash
python train_chronos2_ett.py --config configs/chronos2_ett_hourly.json --device-map cuda
python train_chronos2_ett.py --config configs/chronos2_ett_15min.json --device-map cuda
```

默认配置采用 LoRA、1000 steps、batch size 16、学习率 1e-5、context length 512。第一次建议复制一份配置，把 `num_steps` 改成 50 做 GPU smoke run，并设置全新的输出目录；确认显存、依赖与 API 正常后再跑正式步数。训练脚本拒绝覆盖非空输出目录。

## 5. 数据隔离与发布门禁

每条训练/评估序列按时间顺序划分：

- 0–60%：微调训练窗口。
- 60–70%：Chronos 内部微调验证。
- 70–80%：冻结的候选门禁区间，只用于比较 zero-shot 与微调候选。
- 80–100%：最终测试区间，只做独立报告，不参与候选选择。

候选门禁看整体 MAE、P99、MaxAE、各站点 MAE、P10-P90 覆盖率和区间宽度。通过门禁也不会自动激活；还要经过人工审查、现有模型门禁、影子回放和联合版本登记。当前阈值属于首轮工程阈值，需要结合更多真实设备数据校准。

## 6. Agent 接入

完成训练并生成 `manifest.json` 后，使用：

```bash
python -m diagnostic_agent.cli run-chronos2-oil   --model-dir models/chronos2_ett/hourly   --csv data/ett/ETTh2.csv   --equipment-id ETT-STATION-2   --workspace agent_workspace_chronos2   --device-map cuda
```

Agent 接收预测点值及 P10/P50/P90，并写入共享 EpisodeStore。预测分位数不是故障置信度，也不是 OOD 分数；在未接入独立校准和漂移检测器前，Agent 按保守策略请求人工复核。

## 7. 当前交付环境的实际限制

本次执行容器是 Python 3.13.5、PyTorch 2.10.0+cpu，CUDA 不可用；`chronos`、`peft`、`transformers` 未安装，且运行环境无法解析 `raw.githubusercontent.com`。因此当前完成的是数据下载/校验脚本、Chronos-2 微调编排、模型桥接、门禁、配置和模拟 API 测试；**没有下载 Chronos-2 权重，也没有执行真实 LoRA 微调**。项目包不包含声称训练完成的 Chronos 权重或虚构的精度结果。

当前已核验的真实原始数据只有项目原有 ETTh1.csv：17,420 条小时记录，2016-07-01 00:00 至 2018-06-26 19:00，SHA-256：`f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`。其余三份需要在有网络的训练工作站下载。

**数据规模边界：** ETT-small 是有价值的真实变压器运行基准，但只有两个站点/区域，不是电网级多设备语料。若要把模型作为可泛化的工业油温预测大模型，还需要用户所在单位的多台变压器历史监测数据（油温、负载、电流、电压、环境温度、冷却状态、设备元信息等），并按设备和时间做外部验证。

## 8. 本地代码测试状态

Chronos-2 管线单元测试覆盖输入 schema、时间隔离、尾部误差门禁、预测区间门禁、跨站点评估、Agent Episode 桥接和 LoRA 依赖保护。以上单元测试使用 mock Chronos Pipeline 验证编排逻辑，不等于已经执行真实 Chronos 权重下载或 LoRA 训练。

## 9. 候选的后续对照

GitHub/论文检索另发现 SpanPatchTST（2026 年发表，作者给出 GitHub 实现），其方法是对变压器负荷与油温序列做连续 Patch 掩码自监督预训练后再微调，任务相关性更强。它适合作为下一轮领域专用自监督模型对照；本次没有在隔离环境里完整拉取和审查代码，因此没有将其混入本次可运行包。仓库：<https://github.com/lear-ner97/SpanPatchTST>。

## LoRA 依赖与版本保护

当前依赖下限为 `chronos-forecasting==2.3.2`。训练入口在调用 Chronos-2 `fit()` 前显式检查 `peft` 是否可导入；当配置要求 LoRA 而 PEFT 不存在时会立即报错，不允许 Chronos-2 静默回退到全量微调后仍把运行记录标成 LoRA。请用 `pip show chronos-forecasting peft transformers` 保存实际版本信息。
