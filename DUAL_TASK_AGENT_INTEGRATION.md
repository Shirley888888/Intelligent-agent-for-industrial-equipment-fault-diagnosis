# 双任务模型—Agent 统一接入 V2

## 本次新增

1. 保留 ETTh1 LSTM/TCN 24-step OT 回归预测管线。
2. 新增工业故障多任务 LSTM/TCN：故障类别 logits、可选异常分数、RUL、故障幅值。
3. 新增独立训练/校准数据契约、分类分层发布门禁和不可变 M0 Checkpoint 目录。
4. 新增 `VersionedIndustrialFaultBridge`，安全加载权重并核验 SHA-256。
5. 新增 `TaskAwareModelRouter`，支持 `etth1_forecast`、`industrial_fault_diagnosis` 和 `multi_task_bundle`。
6. 两种任务使用同一个 DiagnosticAgent / EpisodeStore，但不混淆任务标签、输出单位或模型门禁。
7. 只有独立校准集合格后才标记置信度校准成功；没有经过验证的 OOD/漂移信号时，安全守卫请求人工复核。

## ETTh1 预测模型

项目原有模型根目录可直接接入：

```bash
python -m diagnostic_agent.cli run-task --config configs/task_etth1_forecast.example.json --workspace agent_workspace_unified
```

现有 LSTM/TCN 仍输出未来 24 步 OT 回归预测。`fault_type` 仍为 null，Agent 不会将温度变化自动翻译成故障类别。

如果需要训练 ETTh1 TCN：

```bash
python minimal_loop.py simulate --csv ETTh1.csv --out results_gpu_tcn_default --train-config configs/train_tcn.json
```

## 工业故障多任务模型

数据格式见 `INDUSTRIAL_FAULT_DATASET_SCHEMA.md`。训练：

```bash
python -m diagnostic_agent.cli train-industrial-model --dataset path/to/industrial_fault_dataset.npz --config configs/industrial_fault_lstm.example.json --out industrial_fault_lstm_registry
```

TCN 则使用 `configs/industrial_fault_tcn.example.json`。通过门禁时生成 `versions/M0/` 和 `current.json`；未通过时只生成 `candidates/C0/`，禁止推理加载。

运行：

```bash
python -m diagnostic_agent.cli run-task --config configs/task_industrial_fault.example.json --workspace agent_workspace_unified
```

`input_json` 要提供与 checkpoint 输入维度一致的真实窗口，不得用 ETTh1 预测值冒充工业传感器多特征窗口。

## 两种任务一次运行

修改 `configs/task_both.example.json` 中的路径，使其指向你实际的模型根目录和工业推理窗口，然后运行：

```bash
python -m diagnostic_agent.cli run-task --config configs/task_both.example.json --workspace agent_workspace_unified
```

路由器顺序运行两个任务，二者分别写入同一 EpisodeStore。输出保存在多任务结果列表中，不进行跨任务数值融合。

## 测试

```bash
python -m unittest discover -s tests_dual_task -v
python -m unittest discover -s tests_integration -v
python -m unittest discover -s tests_agent -v
python -m unittest discover -s tests_dual -v
python -m unittest -v test_loop test_torch_version_checks
```

测试中的工业故障数据是确定性合成夹具，只用于验证网络结构、训练/加载、门禁、路由和安全行为；不代表真实设备诊断精度。


## 辅助任务发布门禁说明

工业任务可以仅训练故障分类，也可以同时提供异常检测、RUL 和故障幅值标签。只要 NPZ 中包含相应辅助任务标签，配置就必须为该任务设置对应发布阈值：`min_anomaly_auroc`、`max_rul_mae`、`max_amplitude_mae`。若缺少阈值或验证指标未达标，模型只保存为候选，不会激活。阈值应按真实设备和工程单位设定。

独立校准拟合集 `cal_fit` 与校准评估集 `cal_eval` 必须互相独立，且都不得与训练集、模型验证集发生窗口或故障事件泄漏。示例测试使用的合成数据只验证数据管线、网络输出和 Agent 接线，不代表工业现场准确率。
