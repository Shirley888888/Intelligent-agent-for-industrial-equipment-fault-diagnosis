# 双任务统一模型—Agent V2 验收记录

## 验收范围

- 保留 ETTh1 原有 LSTM/TCN 预测任务（OT 未来 24 步回归）。
- 新增面向真实工业标注数据的 LSTM/TCN 多头故障模型训练与版本门禁。
- 两种任务经过 `TaskAwareModelRouter` 进入同一个 `DiagnosticAgent` 和 `EpisodeStore`，但任务输出、单位、校准信息及门禁独立，不做数值融合。
- 工业模型输出故障类别、类别概率；异常分数、RUL 和故障幅值仅在存在对应训练标签时输出。

## 自动化测试

| 测试套件 | 总数 | 通过 | 跳过 | 失败 |
|---|---:|---:|---:|---:|
| `tests_dual_task` | 9 | 9 | 0 | 0 |
| `tests_integration` | 8 | 8 | 0 | 0 |
| `tests_agent` | 29 | 29 | 0 | 0 |
| `tests_dual` | 12 | 12 | 0 | 0 |
| `test_loop` + `test_torch_version_checks` | 29 | 28 | 1 | 0 |
| **合计** | **87** | **86** | **1** | **0** |

唯一跳过项是 CUDA 跨设备一致性测试：当前验收环境使用 CPU-only PyTorch，没有可用 CUDA。Python 编译检查通过。

## CLI 端到端烟测

使用合成、确定性工业窗口运行 `train-industrial-model`，随后用同一份 `multi_task_bundle` 执行项目已有 ETTh1 LSTM M0 和刚生成的工业故障 LSTM，再将两条 Episode 写入同一个 Agent 工作区。共成功写入 2 条 Episode。两个结果均进入 `human_review`，原因是烟测没有注入真实 OOD/漂移检测器；这是预期的 fail-closed 行为。

烟测中的工业数据、宽松门禁阈值只用于验证数据格式、训练入口、Checkpoint、版本加载、Agent 路由与 Episode 持久化，不可视为工业诊断精度或生产门禁阈值。

## 已知边界

1. 随包 ETTh1 的 OT 预测器不等于工业故障分类器。
2. 本次没有用户现场的工业标注数据，因此没有生成真实现场预训练权重，也不能报告真实设备准确率。
3. 真实工业训练必须准备按时间或设备/故障事件隔离的 NPZ，并由数据责任人核实无窗口泄漏。`split_policy` 是声明，不能代替原始设备/事件 ID 审计。
4. 置信度校准要求独立 `cal_fit` 与 `cal_eval`；校准 ID 必须配置一致。
5. 若数据包含异常、RUL、幅值标签，则必须分别配置 AUROC/MAE 发布阈值；缺失阈值会拒绝发布。
6. 尚未接入生产级 OOD 与在线漂移检测器，故其分数缺失时请求人工复核。

## 复测命令

```bash
python -m unittest discover -s tests_dual_task -v
python -m unittest discover -s tests_integration -v
python -m unittest discover -s tests_agent -v
python -m unittest discover -s tests_dual -v
python -m unittest -v test_loop test_torch_version_checks
python -m compileall -q diagnostic_agent dual_evolution tests_agent tests_dual tests_integration tests_dual_task
```
