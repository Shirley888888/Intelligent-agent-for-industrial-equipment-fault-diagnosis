# 工业故障 LSTM/TCN 训练数据格式与双任务接入

本项目现在保留两条独立任务链，并通过同一个 `DiagnosticAgent` 和 `EpisodeStore` 汇合：

- **ETTh1 预测任务**：现有 LSTM/TCN 预测未来 24 个时间步的 OT，属于回归，不产生故障类别。
- **工业故障诊断任务**：新增多头 LSTM/TCN，使用真实工业标注学习故障类别，并可选学习异常分数、RUL 和故障幅值。

不要把 ETTh1 的 OT 目标值伪装成工业故障标签。当前上传包没有真实工业标注数据，因此本包新增的是工业任务的可训练模型结构、数据接口、训练门禁、版本目录和 Agent 接入；没有声称提供真实工业预训练权重。

## 1. NPZ 数据文件结构

训练入口要求一个 `.npz` 文件。必须在数据准备阶段按时间或设备分组，分别提供训练、模型验证、校准拟合集和独立校准评估集。所有窗口应避免跨集合重叠。

| 数组 | 形状 | 必需 | 含义 |
|---|---|---|---|
| `X_train` | `[N_train, input_len, n_features]` | 是 | 训练传感器窗口 |
| `X_val` | `[N_val, input_len, n_features]` | 是 | 模型早停与发布门禁验证窗口 |
| `y_fault_train` | `[N_train]` | 是 | 故障类别整数索引 |
| `y_fault_val` | `[N_val]` | 是 | 验证类别整数索引 |
| `X_cal_fit` | `[N_cal_fit, input_len, n_features]` | 校准时必需 | 温度缩放拟合集，和训练/验证集独立 |
| `y_fault_cal_fit` | `[N_cal_fit]` | 校准时必需 | 校准拟合类别标签 |
| `X_cal_eval` | `[N_cal_eval, input_len, n_features]` | 校准时必需 | 独立校准评估集，不能参与温度拟合 |
| `y_fault_cal_eval` | `[N_cal_eval]` | 校准时必需 | 独立校准评估标签 |
| `y_anomaly_train/val` | `[N]` | 否 | 0/1 异常标签；缺失值可用 NaN |
| `y_rul_hours_train/val` | `[N]` | 否 | RUL 数值；缺失值可用 NaN |
| `y_fault_amplitude_train/val` | `[N]` | 否 | 故障幅值；缺失值可用 NaN |

故障类别索引必须与训练配置 `fault_classes` 的顺序一致，例如 `0=normal, 1=bearing_fault, 2=unbalance_fault`。输入特征顺序、采样频率、单位及窗口长度必须在训练与推理时保持一致。

`split_policy` 是调用者对数据隔离方式的明确声明：`chronological_disjoint` 或 `equipment_group_disjoint`。程序会保存该声明，但无法仅凭 NPZ 数组证明来源是否真的满足隔离；数据负责人仍需核对设备、故障事件和滑动窗口是否重叠。

## 2. 训练与模型门禁

```bash
python -m diagnostic_agent.cli train-industrial-model \
  --dataset path/to/industrial_fault_dataset.npz \
  --config configs/industrial_fault_lstm.example.json \
  --out industrial_fault_lstm_registry
```

配置可以选择 `model_name=LSTM` 或 `model_name=TCN`。当前发布门禁默认同时要求：整体验证准确率达标、每个故障类别召回率达标、每类验证样本量足够，以及独立校准集上的 ECE 与样本量达标。通过后才写入 `current.json` 并激活 M0；未通过时仅保存到 `candidates/C0`，没有激活指针，Agent 无法加载。

若不提供独立校准数组，模型仍可完成训练和验证，但 `confidence_calibrated=false`，默认门禁不会激活该模型。这是为了避免在训练或早停所用验证集上同时拟合和自评置信度。

如果 NPZ 同时提供了某个辅助任务的标签（异常检测、RUL 或故障幅值），发布门禁也必须为该任务配置独立阈值：

- 提供 `y_anomaly_train/val` 时，配置 `min_anomaly_auroc`；
- 提供 `y_rul_hours_train/val` 时，配置 `max_rul_mae`；
- 提供 `y_fault_amplitude_train/val` 时，配置 `max_amplitude_mae`。

未配置对应阈值、验证指标无法计算或指标越界时，该候选不会激活，只保存到 `candidates/C0`。阈值必须依据设备维护容忍度、单位和独立验证数据确定，不要照搬示例中的数值，也不要为了让候选通过而放宽阈值。只训练分类头时可以不提供辅助任务数组；此时相应门禁不参与判定。

## 3. 推理数据格式

`input_json` 内容模板位于 `examples/industrial_fault_input.template.json`。模板中的 `history` 是空数组，必须先替换为真实二维数值数组，形状与 checkpoint 的 `[input_len, n_features]` 完全一致后才能运行：

```json
{
  "episode_id": "PUMP-EP-0001",
  "equipment_id": "PUMP-01",
  "prediction_time": "2026-11-02T10:00:00+08:00",
  "operating_condition": "steady_load",
  "feature_names": ["vibration_x", "vibration_y", "bearing_temp", "motor_current", "pressure", "flow", "speed"],
  "sensor_readings": {"bearing_temp": 68.2, "vibration_x": 2.4},
  "history": [[0.1, 0.2, 68.2, 1.3, 0.5, 2.0, 1500.0]]
}
```

上面 `history` 仅展示行结构；真实文件必须提供完整 `input_len` 行，每行 `n_features` 个数值。

## 4. 双任务统一运行

工业故障任务：

```bash
python -m diagnostic_agent.cli run-task \
  --config configs/task_industrial_fault.example.json \
  --workspace agent_workspace_unified
```

ETTh1 预测任务：

```bash
python -m diagnostic_agent.cli run-task \
  --config configs/task_etth1_forecast.example.json \
  --workspace agent_workspace_unified
```

两条任务分别调用独立的模型桥接器，但使用同一个 Agent 工作区和 EpisodeStore。也可以通过 `configs/task_both.example.json` 的 `multi_task_bundle` 一次依次执行两条任务。运行工业任务前，必须修改 `examples/industrial_fault_input.template.json` 的 `history` 和字段名，并把示例配置中的模型根目录指向门禁通过的实际工业模型注册目录。两个任务的输出 schema、模型版本、训练集和门禁完全独立，不做无意义的温度预测值/故障概率数值融合。

## 5. OOD、漂移与校准约束

工业 checkpoint 的 `confidence_calibrated=true` 只在独立校准拟合集与评估集通过要求后成立，并带有 `calibration_id`。运行环境的 `AGENT_CALIBRATION_ID` 必须与 checkpoint 的 `calibration_id` 完全一致，否则 Agent 将置信度按未校准处理。

当前 CLI 不伪造 OOD 或在线漂移分数，因此未配置外部检测器时，工业诊断会请求人工复核。Python 接入可向 `TaskAwareModelRouter(runtime_signal_provider=...)` 注入经过验证的运行时信号函数，返回 `ood_score` 和 `drift_score`（均需在 `[0,1]`）；不要用固定常数或未经验证的模型置信度代替这两类信号。
