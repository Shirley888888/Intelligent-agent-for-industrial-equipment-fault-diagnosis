# Agent 运行时接入审计：online_loop_minimal111

## 结论

本包增加的 `experiments/agent/` 是 ETTh1 上的 Feedback-Triggered ReAct **策略回放实验**，不是工业故障诊断 LLM Agent 的生产运行时。

## 已有证据

- `experiments/agent/react_trace.jsonl`：2,854 条实验轨迹，主要工具步骤为 `observe_clock`、`release_matured_feedback`、`inspect_recent_errors`、`maybe_train_candidate`、`run_gate`、`publish_or_reject`、`issue_prediction`。
- `experiments/minimal_loop.py` 中 `simulate(..., strategy="react")` 以确定性策略编排反馈释放、监控、候选训练、门禁和版本发布。
- 轨迹里的 `thought` 是代码内固定描述，`tool_input` 只存 SHA-256 摘要；不是实际 LLM Prompt/推理内容或可重放的完整工具参数。
- `tool_catalog.json` 的 `excluded_modules` 明确列出 Risk、Anomaly、RAG、Diagnosis 不在当前独立闭环中。
- 项目 `README.md` 也说明 RAG/Diagnosis 等既有系统模块未包含在本包。

## 因此不能据此宣称已完成的能力

1. 真实 LLM Agent 推理、Prompt 版本迭代或自动 Prompt 修改。
2. 工业设备故障类型识别、剩余寿命/故障幅值与维修诊断的端到端闭环。
3. 实际维修手册/SOP 的 RAG 检索、来源引用及知识更新。
4. 人工纠正、工单结论、工具失败和诊断推翻的线上反馈接入。
5. 真实线上影子流量、灰度发布及 Agent 级自动回滚。

## 当前包完成的整合

- 在原代码上新增独立 `dual_evolution/` V0 扩展层。
- 新增 `EpisodeStore`、分层模型门禁、Agent 冻结案例回放门禁、运行时安全守卫、知识条目审核目录和联合发布登记。
- 新增 `tests_dual/` 与 `configs/dual_evolution_v0.json`。
- 没有将 ETTh1 预测案例伪装成带设备故障标签的真实工业 Episode，也没有把实验轨迹强行转成 Agent 门禁样本。

## 真实 Agent 接入所需的最小上游接口

请在后续接入时提供或补齐以下真实运行时接口：

- `diagnose(input)`: 返回结构化诊断结果和 model version / activation id。
- `agent.run(context)`: 返回 decision、tool_calls（含参数摘要、结果、成功状态）、citations（含 source_id/source_version）、manual_review_requested。
- `feedback.submit(episode_id, reviewer, source_ref, corrected_decision, ground_truth, failure_categories)`：提交工单/专家复核结果。
- 版本接口：返回 prompt_version、policy_version、tool_schema_version、kb_version、retrieval_config_version、risk_policy_version。

这些字段可映射至 `EpisodeStore` 和 `evaluate_agent_candidate`；正式训练标签仍须通过人工或权威业务来源确认。当前不对外部模型 API、真实工具或工单系统作虚构接入。
