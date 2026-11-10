# ATREA 失败驱动的自主研究与模型进化

## 本次增量实现

新增 `dual_evolution/failure_evolution.py`，不替换现有 NAS、模型门禁或联合发布注册表。

- `FailureDrivenEvolution`：串联候选训练、门禁评估、失败诊断、下一候选提案。
- `FailureExperienceLedger`：追加式 JSONL 实验记忆，使用 SHA-256 哈希链检测篡改。
- `diagnose_candidate_failure`：依据训练状态、门禁失败项和证据不足项生成结构化失败标签；明确不把相关性/指标现象伪装成已证实因果。
- `configs/failure_evolution.json`：配置候选族重试次数、全局实验次数、训练时间预算；默认不允许从研究循环直接激活生产模型。

## 状态与安全边界

1. `PASS` 仅返回 `ELIGIBLE_FOR_SHADOW`，意味着可进入独立影子验证，不代表可上线。
2. `REJECT` 和 `REVIEW_REQUIRED` 均不能晋级；证据不足不会被视为通过。
3. 每条实验记录都写入候选配置哈希、训练结果、门禁结果、失败标签、证据和 `production_model_changed=false`。
4. 相同候选配置不重复训练；相同失败签名默认停止当前尝试链，等待新证据或研究假设。
5. 超过实验次数或训练时间预算即停止；失败实验仍保留在账本中。
6. 本模块不调用 `JointReleaseRegistry.activate`，不修改生产模型指针。灰度、审批、激活与回滚仍必须走既有发布流程。

## 接入方式

宿主系统提供三个回调：

- `train_and_evaluate(candidate, attempt)`：训练候选模型并返回状态、指标、训练耗时等。
- `gate_evaluate(candidate, training_result)`：在相同冻结数据切分上比较候选与当前基线，返回 `PASS` / `REJECT` / `REVIEW_REQUIRED`。
- `propose_next(candidate, diagnosis, history)`：根据失败证据与历史实验生成下一候选；不得重复配置。

示意：

```python
from dual_evolution.failure_evolution import (
    FailureDrivenEvolution, FailureExperienceLedger, FailureEvolutionConfig,
)

engine = FailureDrivenEvolution(
    FailureExperienceLedger("agent_workspace/failure_experiments.jsonl"),
    FailureEvolutionConfig(),
)
result = engine.run(
    initial_candidate={"architecture": "transformer", "seed": 11},
    train_and_evaluate=train_candidate,
    gate_evaluate=compare_against_incumbent,
    propose_next=agent_propose_next,
    candidate_family="industrial_oil_temperature",
)
# 只有 ELIGIBLE_FOR_SHADOW 才能进入独立影子验证；不能在这里直接上线。
```

## 重要验证边界

单元测试验证编排、安全边界、去重、预算、异常记录和账本完整性。要证明模型确实优于生产基线，仍需宿主训练器提供同一冻结测试窗口上的真实预测值，并运行正式门禁与影子验证。单元测试通过不等于预测性能提升。

## 并发限制

账本当前采用单写入者设计；多进程/多机并发训练前，应由宿主调度器加锁或迁移到支持事务的数据库。
