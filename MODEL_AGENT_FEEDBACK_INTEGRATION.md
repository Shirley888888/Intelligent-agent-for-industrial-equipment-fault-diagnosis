# LSTM/TCN → Agent → 人工纠正候选：本次接入说明

## 1. 已完成的真实模型接入

- `minimal_loop.ForecastModel` 根据 checkpoint 元数据加载 LSTM 或 TCN。整合包包含原有 LSTM M0，以及本次从 ETTh1 数据训练并登记为 M0 的 TCN 演示版本 `run_tcn_agent_etth1/`（1 epoch，仅验证模型加载与数据/Agent 接口，不应据此判断 TCN 模型精度优劣）。
- `TrainConfig` / `train_candidate_tool` 支持 `torch/LSTM` 与 `torch/TCN`；`configs/train_tcn.json` 给出 TCN 配置。
- `diagnostic_agent.model_bridge.VersionedForecastModelBridge` 从 `VersionManager.current()` 读取激活版本，校验 manifest/checkpoint 后加载模型。
- 模型输入使用最近 96×7 个观测值，输出 24 步 ETTh1 `OT` 回归预测。预测值、目标变量、时间戳、model version、activation_id、checkpoint SHA-256 与数据窗口会进入 Agent 报告/Episode。
- Agent 的故障类别权威仍然属于独立模型。当前 ETTh1 回归器未输出 fault class/confidence，因此桥接器明确留空并触发人工复核，不会把回归曲线转成故障标签。

## 2. 接入真正的故障 LSTM/TCN

当前仓库没有真实设备故障分类/RUL/幅值专用 checkpoint、预处理器、标签映射或校准器。为避免伪造，新增 `StructuredPredictorBridge` 作为模型原生输出适配接口：

```python
import numpy as np
from diagnostic_agent.model_bridge import StructuredPredictorBridge

# 这里必须替换为你自己的已训练模型推理函数；不得从 Agent 生成故障标签。
def predict_fault_model(history: np.ndarray) -> dict:
    # 输出字段应来自真实模型及其已验证校准器。
    return {
        "fault_type": model_fault_label,
        "confidence": calibrated_confidence,
        "confidence_calibrated": True,
        "ood_score": model_ood_score,
        "drift_score": drift_score,
        "rul_hours": predicted_rul,
        "fault_amplitude": predicted_amplitude,
    }

bridge = StructuredPredictorBridge(
    predict_fault_model, model_name="TCN", model_version="F12",
    activation_id=3, checkpoint_sha256="<真实 checkpoint SHA-256>",
)
context = bridge.predict_context(
    history, equipment_id="PUMP-001",
    prediction_time="2026-11-02T10:00:00+08:00",
    operating_condition="steady_load", sensor_readings=sensor_payload,
)
report = agent.run(context)
```

必须由真实模型返回 `fault_type`；置信度只有在对应设备族/任务的校准流程验证后才可标记 `confidence_calibrated=True`。`activation_id` 应使用真实模型发布系统的激活编号。

## 3. 人工纠正如何生成候选

`DiagnosticAgent.submit_feedback()` 在反馈状态变为 `verified` 后自动调用 `StrategyCandidateGenerator`。默认同一失败类别至少有 2 个已核实 Episode 才提案。

| 人工确认的失败类别 | 生成的候选 | 候选内容
|---|---|
| `agent_decision_error` / `agent_overconfidence` / `evidence_or_citation_error` / `workflow_or_escalation_error` | Prompt | 结构化约束补丁及冻结回放断言
| `tool_routing_error` / `tool_execution_error` | 工具路由 | 只读工具顺序、失败升级和受限重试策略
| `retrieval_miss` / `knowledge_outdated_or_conflict` | 检索策略 | 查询字段组合、设备/日期/审核状态过滤、冲突升级规则

输出写入工作目录 `strategy_candidates.jsonl`。只产生候选，不修改当前运行配置；必须用冻结案例回放、Agent Gate 和人工批准后才可进入联合发布。

## 4. 验证

```bash
python -m unittest discover -s tests_agent -v
python -m unittest discover -s tests_dual -v
python -m unittest discover -s tests_integration -v
python -O -m unittest -v test_loop test_torch_version_checks
```

`tests_integration` 覆盖随包真实 LSTM checkpoint→Agent、实际训练的 TCN checkpoint→Agent，以及人工纠正自动产生三类策略候选和候选不直接应用。

## 5. 当前边界

- 本项目 ETTh1 的预测目标是数据集字段 `OT`，其单位按 `dataset_native` 保留，不擅自解释为某一种工业设备的物理量。
- 未提供真实工业设备故障模型 checkpoint、设备预处理参数、故障标签字典、OOD 模型和置信度校准器，因此当前无法声称已验证真实工业故障诊断精度。
- LLM API 不参与故障分类；真实 API 运行还需用户配置兼容服务。
- V0 Episode/JSONL 为单进程单写者原型；生产多进程部署前需加数据库事务/锁、权限与保留策略。
