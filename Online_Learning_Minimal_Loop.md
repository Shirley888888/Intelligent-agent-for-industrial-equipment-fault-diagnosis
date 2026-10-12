# 时间序列预测 Agent 在线学习最小闭环（上传包 v2 修订）

本文对应 96 小时 × 7 变量历史 → 24 小时 OT 预测，MAE 在反标准化后按原始油温量纲计算。新增闭环只管理预测模型；既有风险阈值与知识库保持各自独立的版本。

已核对附件和 GitHub 源码快照 `ab60dbdb07c8d604b8bef13d47ed111874625a7b`。当前系统已有预测、风险、RAG、历史异常和规则诊断能力；Feedback Monitor、Train Candidate、Model Gate 和模型版本管理属于本次新增能力。不能把设计接口写成原系统已经完成的实验。

本次以用户上传的代码包为修改基线，保留其结构。`tool_catalog.json` 由实际函数签名导出规范 I/O，`CHANGELOG.md` 逐项映射修改；包内实际实现的工具以该目录为准，下文 Risk/RAG 等属于原项目外部接入说明。

交付代码是单进程、本地文件方案。NumPy Linear 是可独立运行的闭环参考后端；PyTorch 后端复用仓库的 LSTMModel，保持两层 128 隐藏单元结构。当前环境没有 PyTorch，实跑结果仅属于 NumPy Linear，不是附件中的正式 LSTM 结果。

## 1. Tool 清单与输入输出定义

### 1.1 接口约定

- 特征顺序固定为 `[HUFL, HULL, MUFL, MULL, LUFL, LULL, OT]`，target 为 OT，输入 `float[96,7]`，预测 `float[24]`。
- `prediction_time=t` 定义为本次发布预测的逻辑时刻，同时是首个预测目标时间。输入最后时间为 `t-1h`，目标时间为 `t,...,t+23h`。ETTh1 原始时间没有时区信息，不擅自添加 UTC 标记。
- 日期对外使用 ISO 8601 字符串；内部使用 datetime。MAE、温度、温升为原始油温量纲；小时斜率为该量纲/小时。
- 模型版本与模型类型分开：`model_version="M1"` 是某一不可变产物，`model_name="LSTM"` 是网络结构。
- checkpoint 与归一化参数、特征顺序、结构配置绑定保存，升级和回滚整体切换。
- 输入不合法或计算失败时抛出异常，由调度器保存失败记录并保留现役模型；不输出伪造的 MAE 或默认允许上线。

### 1.2 当前已有模块

当前 Observation 日志中显式调用的是 Forecast、Risk、RAG；Anomaly 和 Diagnosis 是流程内部的独立函数。下表既说明当前代码接口，也说明其 Tool 化边界。

| Tool 或模块 | 当前入口 | 输入参数与类型 | 输出结构与字段类型 | 职责边界 |
|---|---|---|---|---|
| Forecast | `PredictorTool.run(history)` | `history: list[list[float]] / ndarray[96,7]`；初始化参数 `checkpoint: str\|None, scaler_path: str\|None, config_path: str, device: str` | `model: str, input_hours: int, forecast_hours: int, target: str, forecast: list[float](24)` | 当前部署为 LSTM；校验形状、有限值、模型/scaler/config 一致性；数组入口本身不含时间轴 |
| Risk | `risk_tool.analyze(forecast, thresholds=...)` | `forecast: list[float](24), thresholds: dict`；旧接口另有可选 `medium/high: float\|None` | `mean/max/rise/max_slope: float, risk_level: LOW\|MEDIUM\|HIGH, peak_hour_ahead: int, thresholds: dict` | `rise=max(forecast)-forecast[0]`；斜率不含历史末点到预测首点的变化 |
| Anomaly | `anomaly_tool.analyze(history, target_index, thresholds)` | `history: float[96,7]`，也支持一维 OT；`target_index: int=-1, thresholds: dict[str,float]\|None` | `status: NORMAL\|WARNING\|ANOMALY, score: float, latest_temperature: float, rise_6h/rise_12h/rise_24h: float, slope_6h/slope_24h: float, max_abs_jump/max_positive_slope: float, reasons: list[str], thresholds: dict[str,float]` | 检查已观测历史；score 是规则计数归一值，不是异常概率 |
| RAG | `RAGTool.retrieve(query)` | `query: str`；构造配置 `backend: str, top_k: int, similarity_threshold: float, chunk_size: int, chunk_overlap: int`；可选 `tfidf_threshold: float`；Embedding 后端需 `embedding_model: str`。知识正文来自 `DEFAULT_KNOWLEDGE` | `list[{id: str, text: str, score: float, accepted: bool}]` | 保留 Top-K 候选及阈值判断；没有候选通过时不得补造知识依据 |
| Diagnosis | `diagnosis_tool.diagnose(anomaly, risk)` | `anomaly: AnomalyResult, risk: RiskResult` | `level: NORMAL\|WARNING\|ANOMALY, diagnosis/reason/recommendation: str, evidence: list[str]` | 确定性规则融合；不是自由生成或真实故障分类 |

风险阈值的加载与校准是初始化支持函数，不等于 Forecast 的在线学习。现有 `train_gpu.py` 是离线训练入口，会读取完整数据并计算测试结果，不能直接作为在线候选训练 Tool 调用。

### 1.3 本次新增 Tool

| Tool | 输入参数与类型 | 输出结构与字段类型 | 副作用与约束 |
|---|---|---|---|
| 预测封装 `predict_tool` | `model: ForecastModel, history: float[96,7], history_times: list[datetime](96), prediction_time: datetime, model_version: str, activation_id: int, origin_idx: int` | `FeedbackEpisode`，初始 `status="pending", truth=None, MAE=None` | 将预测和版本绑定；输入时间全部 `< prediction_time`；保存原始量纲历史与预测 |
| 反馈归集 `complete_feedback` | `episode: FeedbackEpisode, observed_values: float[T,7], observed_times: list[datetime], as_of: datetime` | 完整 `FeedbackEpisode`，`truth: list[float](24), MAE: float, feedback_time: str, status="complete"` | 仅在 24 个目标全部到达后计算 MAE；按目标时间逐点匹配；原预测不变 |
| Feedback Monitor | `episodes: list[FeedbackEpisode], model_version: str, activation_id: int, baseline_mae: float, as_of: str`；构造参数见第 3 节 | `model_version: str, activation_id/n/count: int, baseline_mae: float, recent_mae: float\|None, threshold: float`（检查时）, `episode_ids: list[str], streak: int`（每次调用）, `trigger_retrain/degraded: bool, checked_count/evaluated_blocks: int, reason/as_of: str` | 只统计当前激活批次完整反馈；维护已检查数量、连续超阈值块数和冷却状态；不直接训练 |
| Train Candidate | `data_version: DataVersion, train_config: TrainConfig, observed_values: float[T,7], observed_times: list[datetime], candidate_dir: str\|Path` | `candidate_id/data_version/checkpoint_path/checkpoint_sha256/trained_as_of: str, validation_metrics: {MAE: float, unit: str, windows/best_epoch: int}, train_config: dict, manifest: dict` | 仅写候选目录；不覆盖 M0；训练与 scaler 拟合都排除门禁历史块；早停只看独立 validation |
| Model Gate | `candidate: CandidateResult, incumbent: ForecastModel, incumbent_version: str, observed_values: float[T,7], observed_times: list[datetime], device: str` | `allow_upgrade/recent_ok/historical_ok: bool, recent_limit/historical_limit: float`；四个 MAE；`candidate_id/incumbent_version/checkpoint_sha256/evaluated_as_of: str`；`recent/historical: {incumbent: EvalResult, candidate: EvalResult}` | 两模型在完全相同近期/历史窗口上重算原始量纲 MAE；只读；不替换历史 Episode |
| 版本管理 `VersionManager` | `publish(candidate: dict, gate: dict\|None, as_of: str, baseline_mae: float)`；`activate(version: str, as_of: str, action: str="rollback")`；`load(version: str)`；`current()` | 发布/回滚：`model_version/action/activated_as_of: str, activation_id: int, baseline_mae: float`；load：`tuple[ForecastModel, manifest: dict]`；current：当前指针字典 | 仅 M0 初始化允许无 gate；升级需检查 gate 绑定的 incumbent、activation_id、候选、数据快照及 hash；发布不可变版本；验证后原子替换指针 |

```python
class EvalResult(TypedDict):
    MAE: float
    windows: int
    target_range: list[int]       # [start, end)，目标时间范围
    origins: list[int]            # 完整窗口的预测起点

@dataclass
class DataVersion:
    data_version: str
    observed_end: int             # 已观测前缀 [0, observed_end)
    as_of: str
    train_ranges: list[tuple[int, int]]
    validation_range: tuple[int, int]
    historical_range: tuple[int, int]
    recent_range: tuple[int, int]
    snapshot_sha256: str          # 只散列已观测前缀
```

论文可写入的接口说明：

> 系统采用结构化工具接口组织预测与反馈处理。预测工具接收具有明确特征顺序和时间边界的历史窗口，输出油温预测序列及其模型版本标识；反馈归集工具在目标观测全部到达后形成不可变的预测—观测对应记录。反馈监控工具依据当前激活模型的近期误差与冻结基线判断是否发起候选训练，训练工具返回独立保存的候选产物与验证指标。模型门禁工具在相同近期窗口和冻结历史窗口上比较候选与现役模型，仅当近期误差改善且历史误差退化未超过容差时允许升级。版本管理工具负责保存模型、归一化参数、训练配置及门禁记录，并通过当前版本指针完成上线与回滚。上述接口将数值计算、效果观测、候选生成和版本切换分离，使反馈驱动的模型更新过程具有明确的数据来源与执行边界。

该段描述新增机制；将其写入论文后，实验章节应单独报告对应验证，不能用原有三案例结果替代在线闭环证据。

## 2. Feedback Episode 数据结构

```python
class FeedbackEpisode(TypedDict):
    episode_id: str
    prediction_time: str          # 首个预测目标时刻，也是发布逻辑时刻
    model_version: str            # 实际生成此 forecast 的版本
    activation_id: int            # 每次升级/回滚递增，隔离反馈统计
    origin_idx: int               # 首个目标的行索引 p
    target_end_idx: int           # p + 24，右开边界
    history: list[list[float]]    # [96,7]，原始量纲
    history_times: list[str]      # 96 个时间戳，全部 < prediction_time
    forecast: list[float]        # 24 个 OT 点预测，原始量纲
    target_times: list[str]      # t,...,t+23h
    truth: list[float] | None    # 24 个匹配真值；尚未收齐为 None
    MAE: float | None            # 未收齐时为 None，不能为 0
    feedback_time: str | None    # 收齐全部目标的逻辑时刻
    status: Literal["pending", "complete"]
```

下面是完整形状的教学示例，不是 ETTh1 实测结果。

```python
from datetime import datetime, timedelta
import numpy as np

t = datetime(2018, 3, 10, 0, 0)
def stamp(x):
    return x.isoformat(timespec="seconds")

history = [[5.0, 2.0, 1.5, 0.4, 4.0, 1.2, 30.0] for _ in range(96)]
forecast = [30.0 + 0.1*i for i in range(24)]
truth = [x + 0.2 for x in forecast]
episode = {
    "episode_id": "example-001",
    "prediction_time": stamp(t),
    "model_version": "M0", "activation_id": 0,
    "origin_idx": 1000, "target_end_idx": 1024,
    "history": history,
    "history_times": [stamp(t-timedelta(hours=96-i)) for i in range(96)],
    "forecast": forecast,
    "target_times": [stamp(t+timedelta(hours=i)) for i in range(24)],
    "truth": truth,
    "MAE": float(np.mean(np.abs(np.array(forecast)-truth))),  # 约 0.2
    "feedback_time": stamp(t+timedelta(hours=24)),
    "status": "complete",
}
```

创建时先保存 pending 记录；到 `t+24h` 时，`t+23h` 的观测已满足严格 `< as_of`，才追加 complete 记录。`predictions.jsonl` 保存发布记录，`feedback.jsonl` 保存完整反馈；用 episode_id 关联。升级不重新标注旧样本的 model_version。

## 3. Feedback Monitor

默认每 24 小时发布一次 24 步预测，目标窗口不重叠；每次完整反馈对应一个 Episode MAE。

```python
N = 30                     # 最近 30 个完整 Episode，日发布对应约 30 天
RELATIVE_THRESHOLD = 0.20  # 比冻结基线恶化 20%
ABSOLUTE_THRESHOLD = 0.10  # 原始油温量纲的最小增加量
PATIENCE = 2               # 连续两个不重叠的 N 样本块超阈值
COOLDOWN_EPISODES = 30      # 两次训练尝试至少间隔 30 个新反馈

recent_mae = mean(e.MAE for e in latest_N_complete_episodes)
trigger_threshold = baseline_mae + max(
    RELATIVE_THRESHOLD * baseline_mae,
    ABSOLUTE_THRESHOLD,
)
degraded = recent_mae > trigger_threshold
trigger_retrain = (
    enough_complete_feedback
    and consecutive_degraded_blocks >= PATIENCE
    and new_feedback_since_last_attempt >= COOLDOWN_EPISODES
)
```

M0 的基线由上线前独立校准块计算，未参加 M0 参数拟合或早停。新版本的初始基线取其上线门禁近期块 MAE 并冻结；这是被选候选的参考误差，不应称为独立无偏测试结果。升级/回滚后清空监控状态，使用新的 activation_id，重新积累完整反馈；回滚恢复该版本保存的基线。拒绝上线或训练失败也消耗冷却期。

```python
# 返回稳定字段：最近 N 次用于展示；不重叠块用于触发判断。
recent_mae = mean(e["MAE"] for e in complete_episodes[-N:]) if count >= N else None
blocks = 0
while count >= checked_count + N:
    block = complete_episodes[checked_count:checked_count+N]
    checked_count += N
    block_mae = mean(e["MAE"] for e in block)
    streak = streak + 1 if block_mae > threshold else 0
    blocks += 1
trigger = blocks > 0 and streak >= patience and count-last_attempt_count >= cooldown
```

默认值是工程起点，不是统计检验或行业标准。ETTh1 回放仅有约 108 个完整日窗口，N=30、patience=2 的响应较慢；调试可用 N=7、patience=2。建议在上线前开发时段比较 N∈{7,14,30}、相对阈值∈{0.10,0.20,0.30}，先固定再回放。若改为每小时发布，N=30 对应 30 次小时发布且目标高度重叠，不能再解释为 30 个独立日样本。

## 4. Train Candidate Tool

最小方案只重训同一 LSTM 结构，不在每次反馈时搜索多个架构。Model Gate 选择的是模型参数版本。参考 NumPy 后端采用完整 96×7 输入的正则线性回归。

```python
@dataclass
class TrainConfig:
    backend: str = "torch"       # runnable reference 可用 "numpy"
    model_name: str = "LSTM"     # NumPy 后端固定 "Linear"
    epochs: int = 10
    patience: int = 3
    lr: float = 0.001
    batch_size: int = 128
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    seed: int = 42
    device: str = "cuda:0"
    train_stride: int = 1
    ridge: float = 1.0            # 仅 NumPy 使用，固定值，不在 gate 上搜索

def train_candidate_tool(
    data_version: DataVersion,
    train_config: TrainConfig,
    observed_values: np.ndarray,  # 只含已观测前缀，不传完整 CSV
    observed_times: list[datetime],
    candidate_dir: str | Path,
    *, bootstrap: bool = False,  # 仅创建初始 M0 使用
) -> dict:
    ...
```

关键训练逻辑如下，交付的 minimal_loop.py 已实现对应代码。

```python
assert max(observed_times) < as_of
assert snapshot_hash(observed_values, observed_times) == manifest.snapshot_sha256
assert train_targets_disjoint_from(validation_targets, old_targets, recent_targets)

# scaler 只拟合 train_ranges 的原始行，排除冻结历史块。
scaler = fit_scaler(concat_rows(observed_values, manifest.train_ranges))
train = make_windows_within_each_segment(manifest.train_ranges, L=96, H=24)
validation = make_windows(manifest.validation_range, L=96, H=24)
# validation 可以使用更早的已观测历史作为输入，但所有目标必须在 validation 内。

model = LSTMModel(input_len=96, n_features=7, pred_len=24,
                  hidden_size=128, num_layers=2, dropout=0.1)
optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
best_mae, bad_epochs = float("inf"), 0
for epoch in range(cfg.epochs):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = mse(model(scaler.transform(x)), scaler.transform_target(y))
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
    model.eval()
    validation_mae = evaluate_original_scale_mae(model, validation, scaler)
    if validation_mae < best_mae:
        best_mae, bad_epochs = validation_mae, 0
        save_checkpoint(model, scaler, cfg, manifest)  # candidate 目录
    else:
        bad_epochs += 1
    if bad_epochs >= cfg.patience:
        break
return {
    "checkpoint_path": str(candidate_checkpoint),
    "validation_metrics": {"MAE": best_mae, "unit": "degC", "windows": len(validation)},
    "data_version": manifest.data_version,
    "checkpoint_sha256": sha256(candidate_checkpoint),
    # 完整实现还返回 candidate_id、train_config、manifest、trained_as_of。
}
```

训练允许在已经完成时间切分的训练窗口中打乱 minibatch；这不改变回放的时间顺序，也不允许窗口跨入 validation/gate 的目标。训练期不读取近期 gate 分数选 epoch，不评估仍在未来的测试尾部，不覆盖现役模型。

## 5. Model Gate 选型规则

设现役模型与候选在同一近期门禁集 R 上的 MAE 为 E_R(old)、E_R(new)，在同一冻结历史集 A 上的 MAE 为 E_A(old)、E_A(new)。默认门禁为：

```python
recent_limit = E_R_old - max(0.05 * E_R_old, 0.01)
historical_limit = E_A_old + max(0.02 * E_A_old, 0.01)

recent_ok = E_R_new < E_R_old and E_R_new <= recent_limit
historical_ok = E_A_new <= historical_limit
allow_upgrade = finite_nonnegative_metrics and recent_ok and historical_ok
```

近期至少改善 5%，并至少减少 0.01 原始量纲；历史允许恶化 2%，低误差情形至少容许 0.01。现役近期误差为零时，不允许同分升级。任一指标非法、样本不完整或 checkpoint 不兼容都拒绝升级。

```python
def model_gate_tool(candidate, incumbent, incumbent_version,
                    observed_values, observed_times, device="cpu"):
    verify_candidate_hash_and_manifest(candidate)
    new = load_candidate_with_its_own_scaler(candidate)
    R = manifest.recent_range
    A = manifest.historical_range
    # old 与 new 使用同一原始输入、同一真值、同一窗口和权重；
    # 各自使用自身随 checkpoint 保存的 scaler。
    er_old = evaluate_mae(incumbent, R)
    er_new = evaluate_mae(new, R)
    ea_old = evaluate_mae(incumbent, A)
    ea_new = evaluate_mae(new, A)
    return model_gate(er_old, er_new, ea_old, ea_new)
```

R 取最近 N 个已完成当前激活批次 Episode 的目标区间。A 上线前冻结，并始终排除于训练和早停。候选只能在 A、R 上各接受一次本轮门禁判断；拒绝后等待新反馈再训练，不针对相同 gate 反复搜索配置。

门禁重算是“当前候选的回顾评价”，不是候选在历史时刻真实发布过的预测。实际在线表现只计算 predictions/feedback 文件中的原始 forecast。历史 A 重复承担模型选择职责，属于回归门禁，不是最终独立盲测集。

## 6. 模型版本管理与回滚

```text
run/
  current.json
  version_events.jsonl
  predictions.jsonl
  feedback.jsonl
  monitor.jsonl
  retrain_events.jsonl
  candidates/
    C0/checkpoint.pt + manifest.json
    C1/checkpoint.pt + manifest.json + gate.json
  versions/
    M0/M0_checkpoint.pt + manifest.json + gate.json
    M1/M1_checkpoint.pt + manifest.json + gate.json
```

NumPy 后端扩展名为 `.npz`。checkpoint 内包含权重、scaler、特征顺序、目标、结构配置及数据版本信息；manifest 记录 hash、父版本、上线时间、基线和验证指标。候选编号 Ck 与上线版本 Mk 分开，被拒绝的候选不获得上线版本。

```python
# 单写者，本地目录无需数据库。
next_version = "M" + str(max(all_existing_version_numbers, default=-1) + 1)
# 回滚 M2 -> M0 后，下一次新发布仍为 M3，不复用 M1/M2。

load_and_validate(candidate_checkpoint, scaler, feature_order, config)
assert gate.allow_upgrade
require(gate.incumbent_version == current.model_version, "stale version")
require(gate.incumbent_activation_id == current.activation_id, "stale activation")
assert gate.candidate_hash == candidate_hash
copy_into_immutable_version_directory(next_version)
write_manifest_and_gate()
load_and_smoke_test(next_version)
# activate() 内：保存日志原始字节长度，先追加事件，最后提交指针。
# 日志部分写入或指针替换失败：恢复日志原长度，原指针不变，抛异常。
append_version_event()
os.replace("current.json.tmp", "current.json")  # 提交点，之后不再执行 I/O
# publish() 激活失败：清理本次新建的未激活目录，保留原版本和候选。
```

`simulate` 在提交前加载并试推候选；`publish` 返回成功后直接更新内存模型句柄，不再次加载 checkpoint。该异常恢复逻辑面向单写者和普通 I/O 异常；未实现断电/进程崩溃时的多文件事务恢复。日志恢复或目录清理自身失败会显式报错，不声明清理成功。

一键回滚：

```bash
python minimal_loop.py rollback --out run_torch --version M0 --device cuda:0
```

```python
def rollback(version: str, as_of: str):
    # 交付实现对应 VersionManager.activate(version, as_of, "rollback")。
    model, manifest = manager.load(version)  # checksum、结构、scaler、试推理
    assert manifest["activated_as_of"] <= as_of  # 模拟时钟不能回滚到尚不存在的版本
    pointer = manager.activate(version, as_of, "rollback")
    monitor.reset()
    # 下次预测读取新 pointer，并加载该版本 model/scaler。
    # 未完成的旧预测仍按发布时版本收集真值，不混入新 activation 的监控块。
    return pointer
```

回滚只切换指针并追加事件，不删历史版本，不改历史预测，不重新训练。CLI 是离线模拟结果的版本切换工具；本方案不提供并发常驻服务或进程恢复。若接入常驻 Agent，在下一次预测前检查 activation_id，并重载版本、清空 Monitor 状态。损坏版本在指针切换前拒绝。

## 7. ETTh1 离线模拟闭环

### 7.1 初始时间划分

下面全部使用 Python 右开区间 `[start,end)`，行号从 0 开始。固定边界是本次闭环实验协议，不根据未来取值调整。

| 用途 | 行区间 | 是否可拟合 M0/早停 |
|---|---|---|
| M0 训练与 scaler | `[0,12194)` | 参数训练和 scaler |
| M0 早停验证 | `[12194,12914)` | 只选训练 epoch |
| 冻结历史门禁 A | `[12914,14087)` | 永不进入训练/scaler/早停 |
| M0 基线校准 | `[14087,14807)` | 不参与 M0 拟合或早停 |
| 顺序观测与预测 | 从 `14807` 开始 | 真值到达后才可作为后续训练来源 |

该方案重新创建具有清晰时间来源的 M0。旧 checkpoint 如果曾用完整验证段选 epoch，就已经看过上表的历史/校准目标；不能仅通过改名字宣称它满足这份切分协议。可以保留旧模型结构，但严格模拟应按新切分训练 M0。此实验与附件原 70/15/15、逐小时滑窗正式比较属于不同协议，指标不并入旧表。

### 7.2 每次重训的切分

设当前已观测行数为 T，最近 N 个完整反馈目标的起点为 r，最后右边界为 q≤T：

```python
recent_gate = (r, q)                  # 已到达真值的最近 N 个完整日窗口
validation = (r - 720, r)             # 30 天，候选早停，不进入参数训练
historical_gate = (12914, 14087)       # 永久冻结
train_segments = [
    (0, 12914),
    (14087, r - 720),                 # 足够长时加入，否则跳过
]
```

训练滑窗在每个 segment 内独立构造，绝不把删除历史块后的两段直接拼成连续时序。每个训练目标必须完全位于 segment；近期目标与早停目标、训练目标均不重叠。历史门禁的行还从 scaler 拟合中排除。

### 7.3 回放顺序

```python
model, current = bootstrap_M0_from_pre_online_data()
monitor = FeedbackMonitor(n=30, patience=2, cooldown_episodes=30)
pending, completed = [], []

for T in range(14807, len(reservoir) + 1):
    as_of = timestamp_of_row(T-1) + timedelta(hours=1)
    observed = reservoir[:T].copy()     # 唯一允许传给 Tool 的数据

    # 1. 先释放真值；只完成最后目标也已经 < as_of 的 Episode。
    for ep in pending_with_target_end_at_or_before(T):
        completed.append(complete_feedback(ep, observed, as_of))

    # 2. 再监控：只统计当前 model_version + activation_id。
    decision = monitor.run(completed, current.version, current.activation_id,
                           current.baseline_mae, as_of)

    # 3. 只在触发后构造快照；训练/早停/两个 gate 的目标分开。
    if decision.trigger_retrain:
        monitor.mark_attempt()          # 失败、拒绝同样进入冷却
        manifest = build_ranges_from_observed_and_mature_feedback()
        candidate = train_candidate_tool(manifest, config, observed, observed_times,
                                         new_candidate_directory)
        gate = model_gate_tool(candidate, model, current.version,
                               observed, observed_times)
        save_candidate_and_gate(gate)
        if gate.allow_upgrade:
            current = versions.publish(candidate, gate, as_of,
                                       baseline_mae=gate.candidate_recent_mae)
            model = versions.load(current.version)
            monitor.reset()

    # 4. 最后发布新预测；不可取 reservoir[T:T+24] 预填 truth。
    if is_daily_prediction_origin(T) and T < len(reservoir):
        ep = predict_tool(model, observed[-96:], observed_times[-96:],
                          prediction_time=as_of,
                          model_version=current.version,
                          activation_id=current.activation_id)
        pending.append(ep)
        append_prediction_record(ep)
```

完整实现使用串行逻辑时钟：一次训练完成后，候选最早只能影响当前 as_of 或之后的新预测，不改过去输出。训练耗时在此次离线回放中视为零；若模拟实际训练延迟，则在到达 available_at 之前继续使用旧模型，且门禁/上线要核对当时现役版本。

### 7.4 运行与已完成验证

```bash
# 参考后端：只需要 NumPy，自动使用随包 ETTh1.csv。
python minimal_loop.py simulate --out run_numpy

# 使用已配置好的 etth1_gpu 环境；复用 LSTMModel 架构。
python minimal_loop.py simulate --out run_torch --backend torch --device cuda:0 --epochs 10

# 工程演练：在第 60 个反馈到达时请求一次候选训练；不会绕过门禁。
python minimal_loop.py simulate --out run_exercise --exercise-at 60

# 13 项可执行测试，使用 NumPy 参考后端。
python test_loop.py
```

每次指定新的 out 目录，避免覆盖已保存实验。NumPy 默认结果：108 个完整 Episode、1 个尚未收齐的 Episode，prequential MAE=1.415614，冻结基线 MAE=1.350505。三次检查块 MAE 为 1.626427、1.384502、1.288091，阈值为 1.620606；只有首块超阈值，未满足连续两块规则，没有自动重训。

工程演练在第 60 个反馈后实际训练一个候选：近期现役 MAE=1.384502、候选=1.398150；历史现役=1.339423、候选=1.300813。历史条件通过，近期条件失败，候选被拒绝。工程演练不属于 Monitor 自然触发结果，也不能证明在线学习提升精度。

原包 6 项测试已扩展为 13 项并在优化模式下通过，原有覆盖包括：时间切分互斥、完整真值延迟到达及时间对齐、未来扰动下前缀结果不变、监控连续块及版本过滤、双条件门禁、版本递增/回滚/损坏产物保护。版本接受分支使用单元测试的受控指标执行，未将它计作 ETTh1 自然升级。PyTorch/LSTM 训练未在当前环境实跑。

## 8. 如何验证没有未来信息泄露

### 8.1 时间对齐和成熟条件

```python
# 在行 p 处发布；当前只有 [0,p) 可观测。
history = raw[p-96:p, :]         # 最后索引 p-1
forecast_targets = range(p, p+24)
assert max(history_times) < prediction_time
assert target_times[0] == prediction_time

# 在 T=p+24 时完整反馈才能释放；所有真值时间严格 < as_of。
assert episode.target_end_idx <= T
assert target_times[-1] < as_of
truth = observed[p:p+24, OT_index]
MAE = mean(abs(forecast - truth))
```

保存目标时间数组，核对 CSV 时间逐点一致。禁止把 `p-1` 的 OT 当作首个 truth，也禁止少一小时或多一小时的偏移。

### 8.2 拟合与选择边界

```python
assert snapshot.last_observation_time < as_of
assert candidate.train_target_end <= candidate.validation_start
assert candidate.validation_end <= recent_gate_start
assert recent_gate_end <= observed_end
assert no_overlap(train_targets, historical_gate_targets)
assert no_overlap(validation_targets, historical_gate_targets)
assert no_overlap(train_targets, recent_gate_targets)
assert no_overlap(validation_targets, recent_gate_targets)
assert scaler_fit_rows_are_train_rows_only
assert candidate_activation_time >= candidate_training_snapshot_time
```

模型参数与 scaler 都只由允许的训练段拟合；早停只使用 validation；两个 gate 仅用于当前时刻的升级决策。离线 reservoir 可以在加载阶段读完整 CSV，但预测、监控、训练与门禁函数只能收到已观测前缀，不能持有 reservoir 或从原始 CSV 路径重新读取未来。训练数据版本 hash 也仅针对已观测前缀计算。

### 8.3 未来扰动测试

```python
cutoff = online_start + 61 * 24
changed = raw.copy()
changed[cutoff:, :] += 10000.0   # 任意修改 cutoff 之后的数值

A = simulate(raw, stop=cutoff, exercise_at=60)
B = simulate(changed, stop=cutoff, exercise_at=60)
assert A.predictions == B.predictions
assert A.feedback == B.feedback
assert A.monitor_decisions == B.monitor_decisions
assert_equal(A.scaler, B.scaler)
assert_equal(A.M0_weights, B.M0_weights)
assert_equal(A.candidate_weights, B.candidate_weights)
```

该测试已在交付代码中执行通过，覆盖初始化、真实候选训练及门禁前的历史输出。测试的是时间因果约束，不能证明数据集是从未被研究者查看的盲测集。

### 8.4 指标记录边界

实际在线 MAE 仅由当时发布且不可修改的 forecast 与后来到达的 truth 计算。门禁重算误差单独存入 gate.json，不回填或替换 Episode。最后未成熟的预测保留 pending，不纳入在线 MAE。历史门禁反复参与选型，因此不能用其分数作为最终独立泛化结论；后续性能验证仍需要提前固定规则后的新时段。


## 9. v2 可执行接口与验证记录

```bash
python minimal_loop.py tools --out my_tool_catalog.json
python minimal_loop.py simulate --out run_custom --train-config configs/train_numpy.json --gate-config configs/gate.json
python minimal_loop.py versions --out run_custom
python minimal_loop.py rollback --out run_custom --version M0
python -O test_loop.py
```

新增返回类型 CandidateResult、ValidationMetrics、MonitorDecision、GateResult、VersionPointer 在 minimal_loop.py 中直接定义；tool_catalog.json 包含各类型完整字段。FeedbackEpisode_example.json 来自本版真实回放，给出完整历史、预测、真值及其时间。

所有运行边界使用 require() 显式校验，文档伪代码中的 assert 表达约束，交付代码不依赖 assert 执行。版本切换之前校验产物；GateResult 新增现役 checkpoint hash、activation_id、数据快照 hash 和冻结容差配置。新 publish 不接受缺少绑定字段或回滚前的旧 gate。训练参数与门禁参数可通过 JSON 和 CLI 配置，run_config.json 记录本次冻结设置。

results_default_v2/results_exercise_v2 是本次代码重新运行结果；原 v1 记录未覆盖。13 项测试新增了优化模式保护、批量反馈与滚动 MAE、非法反馈/配置/切分、时钟与 sidecar 一致性、接口目录及候选训练失败保留现役模型。GPU/LSTM 尚未在当前环境执行，用户可运行 smoke_torch.py 检查自己的训练环境。
