# 最新实际运行包核验

本包当前源码包含20项核心测试和7项Torch测试。新增GPU小前缀记录真实C1通过门禁并发布M1，随后回滚M0；M2/M3为复用C0的受控状态演练。完整核验见 **GPU_ACCEPTANCE_AUDIT.md / GPU_ACCEPTANCE_AUDIT.json**。

用户报告27项运行中26项通过、1项跳过，完整日志未随附件提供；本次独立复跑为20项通过、7项因无PyTorch跳过。保存结果见 audit_rerun_cpu.txt。源码和原运行产物均未修改。下方历史v2说明中的测试数及未执行GPU字段只对应原阶段。

## 可运行入口

```powershell
python -O -m unittest -v test_loop test_torch_version_checks
python smoke_torch.py --device cuda:0 --out run_gpu_version_checks_new --version-checks
```

必须使用新out目录。这个上传版本的受控检查在原注册目录中追加演练事件，最终M3及其基线0.90不代表真实性能优选。M1上线后没有新预测，不能据此声称在线精度收益。

## 历史使用说明

# 时间序列 Agent 在线学习最小闭环：版本切换一致性修复

本次基于最新上传的 `online_loop_minimal.zip` 修复版本切换失败的一致性问题，保留原结构及 GPU checkpoint 元数据归一化修复。详情见 CHANGELOG.md。

- `VersionManager.activate` 先追加日志，最后原子替换 `current.json`，该替换是提交点。提交前发生普通 I/O 异常时，恢复日志原字节长度并清理临时指针；原指针不变。
- `publish` 在激活失败时删除本次新建的未激活版本目录，保留候选及既有版本，支持重试；模拟器在提交前加载候选模型，提交后直接更新内存句柄。
- 适用单进程、单写者；日志恢复或目录清理本身失败会报告异常。没有新增断电/进程崩溃恢复、多写者支持或断点续跑机制。
- `test_results_activation_fix.txt` 与 `verification_activation_fix.json` 是本次验证记录；旧 v1/v2 记录及 `run_torch_gpu_v2` 原样保留。

以下 v2 说明中的“未执行 GPU”指 v2 初次修订时的验证范围。最新上传包已包含用户 GPU 默认回放结果：108 个成熟窗口、1 个未成熟窗口、M0 基线 MAE≈1.586035、前序 MAE≈1.351492、自然重训 0 次。本次修复在 NumPy 后端执行故障测试，未重新运行 GPU。

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
python minimal_loop.py simulate --out run_lstm --train-config configs/train_lstm.json
python minimal_loop.py simulate --out run_lstm_exercise --train-config configs/train_lstm.json --exercise-at 60
python minimal_loop.py versions --out run_lstm_exercise --device cuda:0
python minimal_loop.py rollback --out run_lstm_exercise --version M0 --device cuda:0
```

你的环境需已安装可使用该 GPU 的 PyTorch；requirements.txt 不重复安装或替换 PyTorch。
`smoke_torch.py` 在真实 ETTh1 前缀上执行一次小规模初始化、候选训练、门禁和回滚，保存 torch_smoke.json。它是执行检查，不能作为模型精度实验。
完整 GPU 回放使用原 LSTMModel，默认两层、128 隐藏单元、dropout=0.1。
初次 v2 修订时未执行 PyTorch；本包现含用户 GPU 完整回放、候选拒绝与小前缀真实接受记录。最新核验范围见 GPU_ACCEPTANCE_AUDIT.md。

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
NumPy 后端只支持 Linear，Torch 后端只支持 LSTM，模型类型与版本编号分别保存。
默认 N=30，监控恶化 20% 且至少增加 0.10 ℃，连续两个不重叠块触发；门禁近期至少改善 5% 且至少 0.01 ℃，历史允许增加 max(2% MAE,0.01 ℃)。建议上线前开发段比较 N=7/14/30 与 10%/20%/30% 恶化阈值，固定配置后再回放。

## 5. 本版实际验证

- `test_results_v2.txt`：13 项 unittest 在 `python -O` 下通过，含未来扰动、旧门禁拒绝、损坏产物/时钟倒退保护及训练失败保留现役模型。
- `results_default_v2`：108 个完整日窗口、1 个待成熟窗口，prequential MAE≈1.415614 ℃，0 次自然重训。
- `results_exercise_v2`：第 60 个反馈后实际训练候选 C1；近期 MAE 未改善，双条件门禁拒绝，版本仍为 M0。
- `verification_v2.json`：本版验证摘要与代码/数据校验信息；`cli_smoke_v2.txt`：命令行运行和回滚记录。
- `results_default_v1/results_exercise_v1/test_results.txt`：上传包中的历史记录，未覆盖。

results_*_v1/v2 属于 NumPy Linear 参考后端；run_torch_gpu* 与 run_gpu_version_checks_v1 属于独立协议的 LSTM GPU 执行记录，均不并入原正式模型对比表。受控测试中的升级分数仅验证版本操作，不作为 ETTh1 性能收益。
候选 manifest 的 checkpoint_path 记录运行机器上的绝对路径；打包结果用于审计，移到另一机器后用 VersionManager.load() 读取 versions/ 下相对命名的保存版本。重跑会产生当前机器的有效候选路径。

## 6. 接入既有 Agent

在原预测发布位置读取 `VersionManager.current()`，再调用 `manager.load(version)` 与 `predict_tool(...)`。
Risk 接收 Episode.forecast，RAG 与 Diagnosis 的接口保持原样。实际观测到达后调用 complete_feedback，依次调度 Monitor、Train、Gate 和 publish。
`model_gate_tool(..., incumbent_activation_id=current["activation_id"])` 必须绑定实际激活编号；回滚后旧 gate 不可重用。
常驻进程每次预测前检查 activation_id，变化时加载模型并 reset Monitor；未成熟旧预测仍按发布时的版本收集真值。

本包支持单进程离线时间顺序回放与指定版本恢复，没有生产常驻服务、自动回滚触发策略、断点续跑或真实训练延迟模拟。模型维护不自动修改风险阈值或知识库。
