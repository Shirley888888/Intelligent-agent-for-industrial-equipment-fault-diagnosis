# 时间序列 Agent 在线学习最小闭环：版本切换一致性修复

本次基于最新上传的 `online_loop_minimal.zip` 修复版本切换失败的一致性问题，保留原结构及 GPU checkpoint 元数据归一化修复。详情见 CHANGELOG.md。

- `VersionManager.activate` 先追加日志，最后原子替换 `current.json`，该替换是提交点。提交前发生普通 I/O 异常时，恢复日志原字节长度并清理临时指针；原指针不变。
- `publish` 在激活失败时删除本次新建的未激活版本目录，保留候选及既有版本，支持重试；模拟器在提交前加载候选模型，提交后直接更新内存句柄。
- 适用单进程、单写者；日志恢复或目录清理本身失败会报告异常。没有新增断电/进程崩溃恢复、多写者支持或断点续跑机制。
- `test_results_activation_fix.txt` 与 `verification_activation_fix.json` 是本次验证记录；旧 v1/v2 记录及 `run_torch_gpu_v2` 原样保留。

以下 v2 说明中的“未执行 GPU”指 v2 初次修订时的验证范围。最新上传包已包含用户 GPU 默认回放结果：108 个成熟窗口、1 个未成熟窗口、M0 基线 MAE≈1.586035、前序 MAE≈1.351492、自然重训 0 次。本次修复在 NumPy 后端执行故障测试，未重新运行 GPU。

## 本次 GPU 实际验证更新（2026-10）

在 `etth1_gpu` 环境（Python 3.11.16、torch 2.7.1+cu128、RTX 5060）完成 torch 后端实际运行：

- 修复 `ForecastModel` 加载 `.pt` checkpoint 时的 metadata 规范化：`torch.save` 保留 tuple 而 JSON sidecar 为 list，统一两侧比较语义后 `validate_candidate` 绑定校验通过，且兼容旧 checkpoint。
- 新增 `test_torch_version_checks.py`（torch 后端版本一致性 8 项测试）与 `smoke_torch.py --version-checks`（升级/回滚/陈旧门禁拒绝/损坏检测演练，损坏检查在临时副本上执行，不破坏正式产物）。
- `python -O -m unittest -v test_loop test_torch_version_checks`：27 项测试通过（1 项因 CUDA 可达跳过），记录见 `test_results_gpu.txt`。
- GPU 全量回放与前次结果一致：默认规则 0 次自然重训；`--train-config configs/train_lstm.json --exercise-at 60` 的候选 C1 近期无足够改善，双条件门禁拒绝，版本保持 M0。小前缀 smoke 中候选以真实分数通过门禁升级，属正常门禁行为。
- 受控门禁分数仅验证版本操作，不作为 ETTh1 性能收益。

## 原 v2 使用说明

本版在你上传的 `ETTh1_Online_Learning_Minimal_Loop(1).zip` 上定向修改。
保留 `minimal_loop.py` 中原有类、函数和主要流程；`models.py`、ETTh1 数据及 v1 实验记录原样保留。
改动以 `modify:` 注释标记，逐项说明见 `CHANGELOG.md`，完整七项设计见 `Online_Learning_Minimal_Loop.md`。

## 1. 一次跑通（NumPy，无需 PyTorch）

使用 Python 3.10 或更新版本，解压后进入 `online_loop_minimal` 文件夹：

```bash
python -m pip install -r requirements.txt
python test_loop.py
python minimal_loop.py simulate --out run_numpy
python minimal_loop.py simulate --out run_exercise --exercise-at 60
python minimal_loop.py versions --out run_exercise
python minimal_loop.py rollback --out run_exercise --version M0
```

每次 `simulate` 必须使用新目录，不覆盖已有实验。最后一条命令将 current.json 切回 M0，恢复该版本 scaler/config/基线，递增 activation_id。历史预测不改写。
工程演练请求候选训练，但不修改 MAE，也不绕过门禁；未接受候选时，只有 M0 可回滚。

## 2. 你的 Windows / etth1_gpu / RTX 5060 环境

```bash
conda activate etth1_gpu
python -m pip install -r requirements.txt
python smoke_torch.py --device cuda:0 --out run_gpu_smoke
python smoke_torch.py --device cuda:0 --out run_gpu_version_checks --version-checks
python -O -m unittest -v test_loop test_torch_version_checks
python minimal_loop.py simulate --out run_lstm --train-config configs/train_lstm.json
python minimal_loop.py simulate --out run_lstm_exercise --train-config configs/train_lstm.json --exercise-at 60
python minimal_loop.py versions --out run_lstm_exercise --device cuda:0
python minimal_loop.py rollback --out run_lstm_exercise --version M0 --device cuda:0
```

你的环境需已安装可使用该 GPU 的 PyTorch；requirements.txt 不重复安装或替换 PyTorch。
`smoke_torch.py` 在真实 ETTh1 前缀上执行一次小规模初始化、候选训练、门禁和回滚，保存 torch_smoke.json；`--version-checks` 额外以受控门禁分数演练升级/回滚/陈旧门禁拒绝/损坏检测。它是执行检查，不能作为模型精度实验。
完整 GPU 回放使用原 LSTMModel，默认两层、128 隐藏单元、dropout=0.1。
GPU 分支已在 `etth1_gpu` 环境（RTX 5060、torch 2.7.1+cu128）实际训练验证，记录见 `test_results_gpu.txt` 与本文“本次 GPU 实际验证更新”一节。

## 3. Tool 参数与输出

```bash
python minimal_loop.py tools --out my_tool_catalog.json
```

随包 `tool_catalog.json` 从函数签名与 TypedDict/dataclass 导出；包括输入参数类型、默认值、完整输出字段、构造参数、时间约束和副作用。
它描述包内实际实现的预测、反馈、监控、训练、门禁、发布、回滚、加载与版本查询。
原项目的 Risk/Anomaly/RAG/Diagnosis 未包含在这次上传的包中，文档仅保留其原系统接口作为接入说明。
ForecastModel 是本地 Python 模型句柄；若接 JSON Tool 调用，由调用适配层先从 VersionManager 加载句柄。
`FeedbackEpisode_example.json` 是本版实际回放的一条完整反馈，含 96×7 历史、24 点预测与真实值、逐点时间和 MAE。

## 4. 完整配置入口

```bash
python minimal_loop.py simulate --out run_custom --train-config configs/train_lstm.json --gate-config configs/gate.json --epochs 10 --train-patience 3 --monitor-n 30 --patience 2 --monitor-relative 0.20 --monitor-absolute 0.10 --cooldown-episodes 30
```

JSON 配置先加载，显式命令行参数覆盖同名配置。`--patience` 是连续超阈值块数；`--train-patience` 是训练早停，两者独立。
`--recent-gain/--old-tolerance/--min-recent-gain/--min-old-tolerance` 覆盖门禁容差；`--lr/--batch-size/--train-stride/--hidden-size/--num-layers/--dropout/--ridge/--seed/--weight-decay/--grad-clip` 覆盖训练参数。
NumPy 后端支持 Linear；Torch 后端支持 LSTM 和 TCN，模型类型与版本编号分别保存。TCN 参数由 tcn_channels 与 kernel_size 配置，并写入 checkpoint 元数据。
默认 N=30，监控恶化 20% 且至少增加 0.10 ℃，连续两个不重叠块触发；门禁近期至少改善 5% 且至少 0.01 ℃，历史允许增加 max(2% MAE,0.01 ℃)。建议上线前开发段比较 N=7/14/30 与 10%/20%/30% 恶化阈值，固定配置后再回放。

## 5. 本版实际验证

- `test_results_v2.txt`：13 项 unittest 在 `python -O` 下通过，含未来扰动、旧门禁拒绝、损坏产物/时钟倒退保护及训练失败保留现役模型。
- `test_results_gpu.txt`：`etth1_gpu` 环境（torch 2.7.1+cu128、RTX 5060）`python -O` 下 27 项 unittest 通过，含 torch 后端版本一致性 8 项（1 项因 CUDA 可达跳过）。
- `results_default_v2`：108 个完整日窗口、1 个待成熟窗口，prequential MAE≈1.415614 ℃，0 次自然重训。
- `results_exercise_v2`：第 60 个反馈后实际训练候选 C1；近期 MAE 未改善，双条件门禁拒绝，版本仍为 M0。
- `verification_v2.json`：本版验证摘要与代码/数据校验信息；`cli_smoke_v2.txt`：命令行运行和回滚记录。
- `results_default_v1/results_exercise_v1/test_results.txt`：上传包中的历史记录，未覆盖。

所有实跑预测结果属于 NumPy Linear 参考后端，不是原论文的正式 LSTM/Linear checkpoint 结果。受控测试中的升级分数仅验证版本操作，不作为 ETTh1 性能收益。
候选 manifest 的 checkpoint_path 记录运行机器上的绝对路径；打包结果用于审计，移到另一机器后用 VersionManager.load() 读取 versions/ 下相对命名的保存版本。重跑会产生当前机器的有效候选路径。

## 6. 接入既有 Agent

在原预测发布位置读取 `VersionManager.current()`，再调用 `manager.load(version)` 与 `predict_tool(...)`。
Risk 接收 Episode.forecast，RAG 与 Diagnosis 的接口保持原样。实际观测到达后调用 complete_feedback，依次调度 Monitor、Train、Gate 和 publish。
`model_gate_tool(..., incumbent_activation_id=current["activation_id"])` 必须绑定实际激活编号；回滚后旧 gate 不可重用。
常驻进程每次预测前检查 activation_id，变化时加载模型并 reset Monitor；未成熟旧预测仍按发布时的版本收集真值。

本包支持单进程离线时间顺序回放与指定版本恢复，没有生产常驻服务、自动回滚触发策略、断点续跑或真实训练延迟模拟。模型维护不自动修改风险阈值或知识库。

## 7. 新增工业诊断 Agent（四轮迭代版）

本包新增 `diagnostic_agent/`，与原 ETTh1 模型在线学习回放保持松耦合。Agent 接收上游独立 LSTM/TCN 的结构化预测，不自行识别或修改故障类别；可使用离线 Planner 做可复现测试，也可配置 OpenAI-compatible Chat Completions 服务。

```bash
# Agent 四轮迭代的单元测试
python -m unittest discover -s tests_agent -v

# 双自进化 V0 测试
python -m unittest discover -s tests_dual -v

# 原模型在线学习回归
python -O -m unittest -v test_loop test_torch_version_checks

# 离线端到端演示（演示资料是合成数据，不是实际设备手册）
python -m diagnostic_agent.cli demo --workspace agent_workspace_demo

# 对实际模型预测 JSON 执行 Agent；默认离线 Planner
python -m diagnostic_agent.cli run --input examples/agent/sample_context.json --workspace agent_workspace
```

如需启用 OpenAI-compatible LLM Planner，在环境变量中配置 `OPENAI_API_KEY`、`OPENAI_MODEL`，可选配置 `OPENAI_BASE_URL`，然后追加 `--provider openai`。密钥只从环境读取，不写入 Episode。

### 7.1 真实版本化 LSTM/TCN Checkpoint → Agent

本版将在线学习闭环中的真实 PyTorch ForecastModel 直接接入 Agent。项目中 `results_gpu_lstm_default/` 是完整训练的 LSTM M0，建议作为 ETTh1 预测接入的首选；`run_gpu_smoke/` 仅为快速联调的 smoke checkpoint。`run_tcn_agent_etth1/` 是基于 ETTh1 数据的一轮训练 TCN 演示发布目录，仅用于验证集成链路，不作为精度结论。`models.py` 中的 TCN 已接入 `TrainConfig`、候选训练与 checkpoint 加载。可用 `configs/train_tcn.json` 训练自己的 TCN 候选。

```bash
# 真实版本目录 + ETTh1 观测数据 → LSTM → Agent → Episode
python -m diagnostic_agent.cli run-model --model-root results_gpu_lstm_default --csv ETTh1.csv --equipment-id ETTH1-LSTM-REPLAY --workspace agent_workspace_lstm

# 同一条桥接链路运行 ETTh1 数据训练的 TCN Checkpoint
python -m diagnostic_agent.cli run-model --model-root run_tcn_agent_etth1 --csv ETTh1.csv --equipment-id ETTH1-TCN-REPLAY --workspace agent_workspace_tcn

# TCN 候选训练配置（按已有在线训练流程运行）
python minimal_loop.py simulate --out run_tcn --train-config configs/train_tcn.json --exercise-at 60 --max-episodes 120
```

`run-model` 会读取 `current.json` 指向的激活版本、校验并加载不可变 checkpoint，使用最近 96×7 个已观测点生成 24 步预测，再把预测值、模型版本、activation_id、checkpoint SHA-256 和输入时间范围交给 Agent。报告及 Episode 保留模型输出与来源追溯。

**任务边界非常重要：** 当前随包 ETTh1 LSTM/TCN 是 `OT` 24 步回归预测器，不是故障分类器，也没有可校准的故障置信度。因此桥接器明确将 `fault_type` 留空，不从温度预测推断故障类别；Agent 按 fail-closed 规则请求人工复核。这条链路证明真实 LSTM/TCN checkpoint 与 Agent 已端到端连通，但不代表已经接入真实设备故障分类/RUL/幅值专用模型。后者可通过 `diagnostic_agent.model_bridge.StructuredPredictorBridge` 传入模型原生结构化输出，不应从回归结果伪造。

### 7.2 人工纠正 → Prompt / 工具路由 / RAG 候选

对 Episode 提交 `verified` 人工反馈后，Agent 会按已审核失败类别自动累计证据；同类问题至少达到 2 个已核实案例（可通过 `AGENT_STRATEGY_MIN_SUPPORT` 调整）才生成候选：

```bash
# 提交人工纠正；失败类别可用逗号分隔
python -m diagnostic_agent.cli feedback --workspace agent_workspace_lstm \
  --episode-id EPISODE_ID --reviewer engineer-01 --source-ref workorder://WO-1001 \
  --status verified --ground-truth '{"result":"verified"}' \
  --corrected-decision "工单确认的实际结果" --failure-categories retrieval_miss \
  --note "已核实的检索遗漏原因"

# 查看自动生成的候选
python -m diagnostic_agent.cli strategy-candidates --workspace agent_workspace_lstm
```

候选类型包括 `prompt`、`tool_route`、`retrieval`。每项都记录证据 Episode、基线版本、拟修改策略和回放断言；候选默认 `requires_human_approval=true`、`requires_frozen_replay=true`、`applied_to_live_agent=false`。单个纠正不会直接改写线上 Prompt、工具白名单或检索策略；争议和无效反馈不会触发候选生成。

新增端到端测试：

```bash
python -m unittest discover -s tests_integration -v
```

**注意：** 默认未设置 `AGENT_CALIBRATION_ID`，运行时安全守卫会因没有经过验证的置信度校准标识而请求人工复核。这是预期的 fail-closed 行为。只有在设备族/任务对应的校准阈值经过验证后，才应配置该标识。新增 Agent 的四轮优化过程、输入契约、执行命令和未完成的生产接入事项见 `AGENT_FOUR_ITERATIONS.md`。


## 8. 双任务统一模型—Agent 接入（V2）

本次在四轮 Agent 与模型端到端 V1 上新增两个独立任务分支，并由同一个 `TaskAwareModelRouter` 路由到同一个 `DiagnosticAgent` / `EpisodeStore`：

- `etth1_forecast`：继续使用原 ETTh1 LSTM/TCN 24 步 OT 回归 Checkpoint，不生成故障类别。新增 `configs/train_tcn.json`，可按原在线训练流程训练 ETTh1 TCN。
- `industrial_fault_diagnosis`：新增多头 LSTM/TCN，使用真实工业标注学习故障类别，并可选学习异常分数、RUL 和幅值；训练数据必须按时间或设备分组隔离。

工业数据数组格式、分层发布门禁、独立校准集、输入示例和安全说明见 `INDUSTRIAL_FAULT_DATASET_SCHEMA.md`。工业训练需要真实标注 NPZ，不使用 ETTh1 的 OT 值制造故障标签。

```bash
# 工业故障模型（LSTM/TCN 选择在 JSON 配置内）
python -m diagnostic_agent.cli train-industrial-model --dataset path/to/industrial_fault_dataset.npz --config configs/industrial_fault_lstm.example.json --out industrial_fault_lstm_registry

# 单任务或双任务统一入口
python -m diagnostic_agent.cli run-task --config configs/task_etth1_forecast.example.json --workspace agent_workspace_unified
python -m diagnostic_agent.cli run-task --config configs/task_industrial_fault.example.json --workspace agent_workspace_unified
python -m diagnostic_agent.cli run-task --config configs/task_both.example.json --workspace agent_workspace_unified

# 新增双任务测试
python -m unittest discover -s tests_dual_task -v
```

`configs/task_both.example.json` 是多任务执行模板，需要先将其中工业模型根目录替换为训练通过门禁的真实模型目录，并将 `examples/industrial_fault_input.template.json` 中 `history` 填成实际的 `[input_len, n_features]` 数值窗口。模板中的空 `history` 不能直接推理，避免把虚构传感器值当作现场数据。两个模型各自保留任务输出、单位、校准与门禁，不会把 OT 预测和故障概率混合。工业故障模型没有真实 OOD/漂移检测器时，运行时守卫继续转人工复核；不得填写人为常数冒充检测信号。

本次开发测试使用合成工业窗口仅验证模型训练接口、Checkpoint 加载、门禁和 Agent 连通，不代表真实工业故障诊断精度。


> 工业故障模型的训练数据和发布门禁详见 `INDUSTRIAL_FAULT_DATASET_SCHEMA.md`。若提供 RUL、故障幅值或异常检测标签，必须为每项配置相应的 MAE/AUROC 发布阈值；未配置或未通过时只保存候选，不生成激活版本。

## Real transformer oil-temperature corpus (ETT-small)

This project now includes a source-verifying downloader and a multi-series LSTM/TCN training path for the official ETT-small oil-temperature dataset. It is distinct from the industrial fault-classification task: the target is `OT` (oil temperature), a 24-step regression forecast.

1. Download the official unchanged CSVs: `python download_ett.py --output-dir data/ett`.
2. Train hourly models on ETTh1+ETTh2: `python train_ett_corpus.py --data-dir data/ett --frequency hourly --model both --out-dir models/ett_corpus --device cuda:0`.
3. Train 15-minute models on ETTm1+ETTm2 separately: use `--frequency 15min`.
4. Send a trained model through the shared Agent: `python -m diagnostic_agent.cli run-ett-corpus --model-dir models/ett_corpus/hourly/LSTM --csv data/ett/ETTh2.csv --equipment-id ETT-STATION-2 --workspace agent_workspace_oil_temp --device cuda:0`.

See `ETT_OIL_TRAINING_GUIDE.md` and `ETT_DATA_LICENSE.md` before training or redistributing artifacts. The two frequency groups must not be concatenated. ETT-small contains two transformer stations and is a multi-site research corpus, not a large fleet-scale pretraining dataset.


## Transformer oil-temperature time-series foundation model (Chronos-2)

The project includes a Chronos-2 LoRA/full fine-tuning pipeline for industrial transformer oil-temperature forecasting. Chronos-2 is a public pretrained time-series foundation model (120M parameters in the upstream model card); this project performs domain adaptation, not pretraining from scratch. The target is OT; six load channels are historical-only covariates. Hourly ETTh1/ETTh2 and 15-minute ETTm1/ETTm2 are separate runs. The downloader pins the upstream ETTDataset commit and verifies the Git blob SHA-1 of each original CSV. Run `python download_ett.py --output-dir data/ett`, install optional packages with `pip install -r requirements_chronos.txt`, run `--preflight-only`, then use the configs in `configs/chronos2_ett_hourly.json` and `configs/chronos2_ett_15min.json` or `run_chronos2_oil_foundation_training_windows.bat`. The supplied starting configuration uses micro-batch 4 and gradient accumulation 4 for an 8GB-class GPU; adjust down if needed. A fine-tuned candidate can be called through `python -m diagnostic_agent.cli run-chronos2-oil --model-dir models/chronos2_ett/hourly --csv data/ett/ETTh2.csv --equipment-id ETT-STATION-2 --device-map cuda`.

Chronos-2 candidate selection uses a 60/10/10/20 chronological split: training, internal fine-tune validation, reserved gate validation and untouched final test. The gate checks aggregate MAE/P95/P99/MaxAE, per-station MAE and probabilistic interval coverage/width. Passing it does not auto-activate the model. See `CHRONOS2_OIL_TEMPERATURE_FOUNDATION_MODEL.md` and `CHRONOS2_EXECUTION_STATUS.md` for source, licensing and execution status. **The build environment used for this integration has no outbound DNS/network, no Chronos/Hugging Face model cache, no `chronos`/`transformers`/`peft`/`accelerate` packages and CPU-only PyTorch**, so Chronos-2 weights and remaining datasets could not be downloaded and real LoRA training was not executed here. The package contains the runnable fine-tuning pipeline and mocked interface tests, not a trained Chronos checkpoint or performance claim. ETT-small has only two published station series and is not a fleet-scale corpus.


## ATREA 自主科研与模型发现升级层

请阅读 `README_ATREA_CN.md`。新增 `atrea/` 目录提供训练触发器、arXiv/Semantic Scholar 论文发现、SQLite 双层经验库、模型注册适配器、统计评估和论文式摘要生成。原有在线学习和 Chronos-2 训练链路保持不变。


## 9. ATREA 自主研究型 Agent 补全（本次新增）

详见 `AUDIT_REPORT_CN.md`（原始功能审计）与 `ATREA_AUTONOMY_COMPLETION.md`（新增能力和接入说明）。

新增每日北京时间 08:00 调度器：
```bash
python -m atrea.scheduler --callback your_package.daily_job:run
```
新增三臂消融运行接口：`atrea.ablation.run_three_arm_ablation`。注意：实际模型训练、论文多源检索、候选门禁与版本激活仍需由宿主项目提供真实回调；新增代码不会伪造训练结果或绕过模型发布门禁。

## ATREA 自主研究升级（V2）

本版本增加每日研究调度、跨来源论文候选检索、A3 独立实验经验库、可审计 ReAct 事件轨迹、逐样本残差诊断，以及调用本仓库 `train_chronos2_ett.py` 的 Chronos-2 真实训练适配器。详细说明见 `ATREA_AUTONOMY_V2.md`。

注意：Chronos-2 适配器支持同一模型家族内的训练配置搜索与微调，不会自动改写 Chronos-2 网络结构。A2/A3 架构消融必须接入真正能训练不同模型架构的适配器；在此之前不能把参数调优报告成架构进化。自动生产激活仍默认关闭。
