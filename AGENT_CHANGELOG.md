# Agent 四轮迭代变更记录

## Iteration 1 — Agent 基础执行链
- 新增 `diagnostic_agent.agent.DiagnosticAgent`：结构化上下文、Planner 接口、工具循环、诊断报告和 Episode 写入。
- 新增离线确定性 Planner 与 OpenAI-compatible Chat Completions 适配器。
- 故障类别只复制上游独立模型的 `model_prediction.fault_type`，LLM 输出的类别字段不会进入诊断结果。
- 新增基础端到端及输入合同测试。

## Iteration 2 — 工具调用可靠性
- 工具执行改为显式 allowlist；只开放已审核知识检索与传感器统计两个只读工具。
- 严格校验参数、调用次数、设备知识域；设备和时间范围来自可信上下文，不接受 Planner 覆盖。
- 增加工具审计 JSONL，错误按工具级隔离并保存。
- 增加 Provider 超时/异常处理、兼容接口归一化和越权工具测试。

## Iteration 3 — 安全策略与证据约束
- 接入 `runtime_safety_decision`：OOD、漂移、置信度未校准、校准 ID 缺失或输入异常都会保守处理/转人工。
- 增加人工复核队列；不向 Agent 暴露设备控制、Shell、SQL、任意网络或文件操作工具。
- 缩窄进入 LLM 上下文的预测字段；LLM 无法改写故障标签。
- 维修检查建议必须绑定本次真实检索到的已审核文档 ID，否则丢弃。
- 增加 OOD、置信度、缺失预测、注入文本和伪造引用测试。

## Iteration 4 — 经验回流与验收
- 每次诊断自动写入 `EpisodeStore`，保留模型激活、策略/Prompt/工具/知识/风险版本及 Prompt/工具模式哈希。
- 根据工具失败、检索未命中、模型不确定和证据引用错误生成失败类别。
- 增加人工反馈入口，要求 reviewer 与 source_ref，并沿用 Episode 标签状态转换。
- 使用冻结案例 `evaluate_agent_candidate` 对真实 Agent 产出进行验收；不使用 LLM 自评。
- 增加账本篡改检测、服务超时、无最终报告和完整 CLI 演示。

## 本次边界

四轮迭代完成的是可运行 Agent 原型，不是生产部署。上传代码包未包含真实设备推理服务、厂家手册/SOP 正式语料、维修工单 API、真实 LLM 凭证和现场校准报告，因此这些适配点仍需现场系统提供。任何演示知识与演示校准标识都不能直接用于实际设备。

## 2026-11 Model–Agent end-to-end integration

- Added active-version LSTM/TCN checkpoint bridge from `VersionManager` to `DiagnosticAgent`.
- Added PyTorch TCN training and checkpoint loading alongside backward-compatible LSTM loading.
- Added a bundled ETTh1-trained TCN M0 integration checkpoint and `run-model` CLI path.
- Added structured adapter contract for device-specific fault-model outputs.
- Verified-human feedback now proposes Prompt, tool-route and retrieval candidates after configurable minimum support; proposals are not applied live.
- Added integration tests and verification report. See `MODEL_AGENT_FEEDBACK_INTEGRATION.md`.
