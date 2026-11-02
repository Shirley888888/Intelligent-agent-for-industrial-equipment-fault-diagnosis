# V0 验收记录

执行环境：当前容器 Python；NumPy 已安装；没有可用 CUDA 设备。

## 测试结果

| 测试集 | 结果 | 说明 |
|---|---:|---|
| 原模型闭环 + Torch 版本检查 | 29 项：28 通过、1 跳过 | CUDA 跨设备一致性测试因无可用 CUDA 跳过；其他测试通过 |
| 双自进化 V0 扩展 | 12 项全部通过 | Episode 哈希链/状态流转、数据分层门禁、关键样本不足、Agent 回放、运行时拒识、知识审核、联合发布、回滚与演示 |
| Python 编译检查 | 通过 | `dual_evolution/*.py` 与入口脚本编译通过 |

## 演示验收结果

合成演示输出位于单独的 `dual_evolution_demo_v0/` 目录：

- 模型门禁：`REJECT`。整体 MAE 从 0.475 降至 0.0792，但 `bearing_fault` 分层退化。
- Agent 回放：`PASS`。一条冻结关键场景满足显式决策、工具、引用与人工升级预期。
- Runtime Guard：`human_review`。置信度数值虽高，但未标记为已校准，因此禁止确定性诊断和自动控制。
- 知识库：草稿检索 0 条；审核通过后检索 1 条。
- 联合发布：候选已登记但未激活，因为模型门禁拒绝。

以上演示数据均为合成数据，只证明代码路径和门禁规则按预期运行，不代表真实设备性能、真实 OOD 检测效果或维修知识正确性。

## 重现命令

```bash
python run_dual_evolution_v0.py --out dual_evolution_demo
python -O -m unittest -v test_loop test_torch_version_checks
python -O -m unittest discover -s tests_dual -v
python -m py_compile dual_evolution/*.py run_dual_evolution_v0.py
```

## 最新上传包 online_loop_minimal111 合并复验（2026-11-02）

本次将 V0 扩展层叠加到用户最新上传的 `online_loop_minimal111.zip`。模型闭环核心 `minimal_loop.py` 与 `test_loop.py` 和此前 V0 基线逐字节一致，未修改核心训练/发布逻辑。

- `python -O -m unittest discover -s tests_dual -v`：12 项通过。
- `python -O -m unittest -v test_loop test_torch_version_checks`：29 项通过，其中 1 项因当前容器无可用 CUDA 而跳过。
- `python -m py_compile dual_evolution/*.py run_dual_evolution_v0.py`：通过。
- `experiments/agent/` 的 ReAct 数据仍按实验回放处理，没有伪装成真实工业诊断 Agent，也没有转换成真实故障标签。

本次复验只证明新增 V0 代码可与该包的现有代码共存；不代表已接入 LLM Agent、RAG、生产影子流量或真实设备数据。
