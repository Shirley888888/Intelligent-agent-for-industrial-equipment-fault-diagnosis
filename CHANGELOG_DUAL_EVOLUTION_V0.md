# 双自进化 V0 增量变更

本次没有改写原 `minimal_loop.py`、`models.py`、ETTh1 数据和既有实验结果，只新增独立扩展层。

新增内容：
- `dual_evolution/episode_store.py`：统一 Episode、审计哈希链、人工标签状态机、失败归因和正负反馈。
- `dual_evolution/model_gate.py`：全局 MAE/P95/P99/MaxAE 与设备、工况、故障类型分层比较。
- `dual_evolution/agent_gate.py`：Agent 冻结案例评测及候选/现役版本对比。
- `dual_evolution/runtime_guard.py`：基于独立 OOD/漂移/置信度信号的确定性安全决策。
- `dual_evolution/knowledge_catalog.py`：知识版本、审核状态、适用设备和有效期治理，TF-IDF 检索基线。
- `dual_evolution/release_registry.py`：模型与 Agent 联合版本清单、发布门禁、人工审批和回滚。
- `dual_evolution/demo_v0.py`、`run_dual_evolution_v0.py`：合成数据端到端演示。
- `tests_dual/`：12 项扩展测试。
- `configs/dual_evolution_v0.json`：门禁起始配置模板。
- `README_DUAL_EVOLUTION_V0.md`：接口、边界及后续接入说明。
- `V0_VERIFICATION.md`：验收记录。

本版未实现自动重训练触发、自动 Prompt 生成、Embedding 检索、真实 OOD 检测、线上流量镜像和生产级多进程服务。所有数据和候选发布仍按人工审核的 V0 规则执行。


## Model–Agent End-to-End Integration (2026-11)

- Added `VersionedForecastModelBridge`: loads the active immutable LSTM/TCN checkpoint from `VersionManager`, predicts from the last 96×7 observed rows, and forwards the 24-step regression forecast plus checkpoint/version/input provenance to `DiagnosticAgent`.
- Extended the PyTorch online training/checkpoint loader to support the repository `TCNModel`, with `configs/train_tcn.json`; the existing LSTM path and historical checkpoint metadata remain backward compatible.
- Added `StructuredPredictorBridge` as the explicit adapter contract for a future device-specific fault LSTM/TCN whose own inference function returns calibrated structured diagnostics.
- Added verified-human-feedback-driven strategy proposals for Prompt, read-only tool routing, and retrieval. Candidates require at least two verified episodes by default and are stored in `strategy_candidates.jsonl`; no proposal is applied to the live Agent.
- Added `run-model` and `strategy-candidates` CLI commands and six integration tests covering a bundled LSTM checkpoint, a freshly trained TCN checkpoint, auto candidate generation, disputed feedback exclusion, support threshold and idempotence.
- Important scope boundary: bundled ETTh1 LSTM/TCN models are 24-step OT regressors, not equipment fault classifiers. They pass forecasts through to the Agent but intentionally produce no fault label; absent a validated fault classifier/calibrator, the Agent requests human review.
