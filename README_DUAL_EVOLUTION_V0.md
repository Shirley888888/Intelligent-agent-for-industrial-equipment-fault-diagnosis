# 双自进化 V0：人工闭环可运行底座

本目录是现有 `online_loop_minimal` 的增量扩展，不重写原来的 `minimal_loop.py`、LSTM/Linear 模型和版本管理。V0 的目标是先把失败记录、人工真值、候选评测、知识审核和联合发布约束做成可审计流程。

## 1. V0 已实现

| 组件 | 文件 | 解决的具体问题 | 当前边界 |
|---|---|---|---|
| 统一经验账本 | `dual_evolution/episode_store.py` | 防止模型、Agent、工具调用和人工纠正分散在不同日志，无法还原一次诊断 | JSONL 追加式、哈希链、单写者；尚非生产数据库 |
| 反馈状态与失败归因 | `episode_store.py` | 防止未经核实的负反馈直接进入训练集 | `pending/verified/disputed/invalid` 状态机、正负反馈与失败分类；V0 人工审核 |
| 分层回归门禁 | `dual_evolution/model_gate.py` | 防止整体 MAE 掩盖设备/工况/故障类型中的局部退化 | MAE、P95、P99、MaxAE；支持分层；需真实元数据及校准阈值 |
| Agent 场景回放门禁 | `dual_evolution/agent_gate.py` | 防止 Prompt 变得流畅但工具、引用或人工升级行为退化 | 对外部 Agent 回放结果做确定性检查，并比较候选与现役版本；本版不执行 LLM |
| 运行时安全控制器 | `dual_evolution/runtime_guard.py` | 防止 OOD、漂移或不可信置信度情况下 Agent 继续确定性诊断 | 消费外部检测信号；本版不训练/计算 OOD 检测器 |
| 知识条目治理与检索 | `dual_evolution/knowledge_catalog.py` | 防止草稿、过期或未审核知识进入 RAG | TF-IDF 词项检索基线；没有 Embedding，待真实语料接入后再加 |
| 联合版本登记 | `dual_evolution/release_registry.py` | 防止模型、Prompt、工具、知识库版本不一致，导致无法复现 | 单写者 JSONL；激活要求模型门禁、Agent 门禁、影子状态和安全审批全通过 |

## 2. 运行

仅需现有 `requirements.txt` 中的 NumPy。Python 3.10+：

```bash
# 在 online_loop_minimal 目录下
python run_dual_evolution_v0.py --out dual_evolution_demo
python -m unittest discover -s tests_dual -v
```

演示使用合成数据，只用于验证机制，不代表任何真实工业设备精度或维修结论。输出在 `dual_evolution_demo/`：

- `experience/episodes.jsonl`：一次诊断记录及人工核验事件；
- `knowledge/catalog.jsonl`：草稿知识经审批后才可检索；
- `registry/`：联合发布登记；
- `demo_report.json`：门禁与安全策略演示结果。

演示特意构造一个反例：整体 MAE、P95、P99 和 MaxAE 均改善，但 `bearing_fault` 分层 MAE 退化，因此模型候选被拒绝。该反例用于验证门禁逻辑，不是实际实验数据。

## 3. 与现有模型闭环如何连接

现有模型侧保持不变。每次 `predict_tool` 产生预测后，Agent 执行完工具调用并形成诊断结果时，由适配层组装一个 Episode，再调用：

```python
from dual_evolution.episode_store import EpisodeStore

store = EpisodeStore("runtime/experience/episodes.jsonl")
store.append_episode(episode_dict)  # label.status 必须从 pending 开始
```

维修工单、拆检结论或专家复核成熟后，再调用：

```python
store.review_episode(
    episode_id="E-2026-0001",
    status="verified",              # 或 disputed / invalid
    reviewer="维修专家工号/角色",
    source_ref="工单号或报告编号",
    ground_truth={"fault_type": "bearing_fault"},
    failure_categories=["model_prediction_error", "retrieval_miss"],
    corrected_decision="人工核实后的结论",
    feedback_signals={"human_rating": -1, "diagnosis_overturned": True},
)
```

推荐记录：`equipment_id`、`operating_condition`、`fault_type`、模型版本和 activation_id、输入窗口指纹、Agent policy/prompt 版本、工具名称与成功状态、引用来源 ID/版本、人工真值来源及失败归因。敏感原始波形/图片可另存对象存储，只在 Episode 留内容哈希和受控引用。

## 4. 门禁使用原则

`evaluate_model_candidate()` 要求 incumbent 与 candidate 在同一冻结样本上评估。没有设备、工况、故障类型元数据的 ETTh1 数据不能证明这些维度上的分层安全性。

- P95 样本数不足时返回 `REVIEW_REQUIRED`，不当作通过。
- P99 需要达到配置的最小样本数；默认 100 只是工程起始值，不是统计学通用定理。
- 关键故障类型样本不足时不能静默 PASS。
- MaxAE 同时比较相对 incumbent 的变化；绝对误差红线 `absolute_max_error_limit` 必须由设备风险和物理量纲确定，不能照抄模板。
- 当前门禁用于回归任务。异常检测还需增加漏报/误报约束；RUL 还需增加危险方向误差约束。

## 5. 安全边界

`runtime_safety_decision()` 不计算 OOD、漂移或置信度，只消费独立检测器输出。默认配置的校准 ID 为 `UNSET`，因此默认不会放行正常诊断。接入真实业务时必须完成按设备/工况的校准并登记 `calibration_id`。即使数值置信度很高，只要它未经校准，也会进入人工复核。

V0 不自动修改模型权重、Prompt、工具路由或正式知识条目；也不允许 LLM 自评后直接通过 Agent 门禁。候选更新仍由人批准。联合发布清单要求模型门禁、Agent 门禁、影子验证和安全策略审批均为 PASS，并在清单中记录对应的证据引用；回滚目标必须是曾经激活过的版本。

## 6. 当前未实现 / 下一轮再做

1. 真实工业数据接入、事件去重、时钟同步、权限控制、多写者并发和服务化部署。
2. 真实 OOD/漂移检测器与阈值校准，及基于成熟标签的在线告警质量监测。
3. LLM Agent 运行时适配器、实际工具执行轨迹采集、Prompt/路由候选生成与冻结案例回放自动执行。
4. 设备手册/SOP 的批量解析、版本冲突处理、Embedding 混合检索、重排与引用定位。
5. 生产环境影子流量、灰度分流、联合回滚编排及崩溃恢复。

完成上述接入前，本版应被视为 **V0 研究原型与验收骨架**，不是可直接控制工业设备的生产系统。
