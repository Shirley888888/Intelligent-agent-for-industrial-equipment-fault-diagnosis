# 模型—Agent 端到端接入验收记录

## 验收结论

- 真实版本化 LSTM checkpoint（`run_gpu_smoke/versions/M0`）经 `VersionedForecastModelBridge` 输出 24 步预测，并进入 Agent 与 EpisodeStore。
- 随包 TCN checkpoint（`run_tcn_agent_etth1/versions/M0`）由 ETTh1 数据训练 1 epoch 后登记，再通过相同桥接器进入 Agent。训练轮数较少，该 checkpoint 只用于验证 TCN 加载/推理/Agent 数据链路，不用于精度结论。
- 对于这两个 ETTh1 模型，因为任务是 `OT` 回归预测，没有故障类别/校准置信度，所以 Agent 保持 `fault_type=null` 并请求人工复核。没有把回归预测转换成故障类别。
- `StructuredPredictorBridge` 已用测试验证可将真实故障模型的结构化输出原样传给 Agent。
- 经 `verified` 的人工纠正达到同类最少 2 个案例后，自动生成 Prompt、工具路由和检索策略候选。候选只写入 `strategy_candidates.jsonl`，要求冻结案例回放及人工批准，不会直接修改线上行为。

## 测试结果

| 测试套件 | 总数 | 通过 | 失败 | 跳过 |
|---|---:|---:|---:|---:|
| Agent 单元测试 | 29 | 29 | 0 | 0 |
| 双自进化 V0 | 12 | 12 | 0 | 0 |
| 模型—Agent / 反馈候选集成 | 8 | 8 | 0 | 0 |
| 原模型闭环 + Torch 版本测试 | 29 | 28 | 0 | 1 |
| **合计** | **78** | **77** | **0** | **1** |

唯一跳过项是 CUDA 跨设备一致性检查；运行环境没有可用 CUDA。所有 CPU 模型加载、训练、预测及 Agent 桥接测试均已执行。

## 环境

- Python：3.13.5
- PyTorch：2.10.0+cpu
- CUDA：不可用
- LLM：端到端使用离线 Planner；OpenAI-compatible 适配器通过 Mock HTTP 测试，没有调用真实在线 API。

## 重点测试

1. `tests_integration/test_model_agent_e2e.py`：捆绑 LSTM checkpoint、ETTh1 训练 TCN checkpoint、测试中训练 TCN checkpoint，以及结构化故障模型输出适配。
2. `tests_integration/test_strategy_candidate_generation.py`：验证 Prompt / tool_route / retrieval 候选生成、最小支持数、争议反馈过滤与幂等性。
3. `test_loop.py`、`test_torch_version_checks.py`：原模型在线学习及版本发布回归。

## 生产接入边界

当前包没有用户真实设备故障分类/RUL/幅值专用 checkpoint、其预处理器、故障标签字典和校准器。请将这些真实产物通过 `StructuredPredictorBridge` 的原生输出接入；不可将 ETTh1 的 OT 回归器当作故障分类器。
