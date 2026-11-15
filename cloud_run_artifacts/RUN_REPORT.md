# ATREA 云端实验运行报告

- 运行日期：2026-11-15（UTC+8）
- 运行环境：托管云端容器，Python 3.13.5，PyTorch 2.10.0+cpu
- CPU：3 vCPU；系统内存约 5.8 GiB
- CUDA：不可用（torch.cuda.is_available() = False）
- 数据：项目随附 ETTh1.csv（只读输入）

## 已完成运行

### 自动化测试
- `tests_integration_cloud.log`：运行 8 项，跳过 0 项，耗时 0.929 秒，失败标记=False
- `tests_dual_task_cloud.log`：运行 9 项，跳过 0 项，耗时 1.82 秒，失败标记=False
- `atrea_synthetic_demo.log`：运行 None 项，跳过 0 项，耗时 None 秒，失败标记=True
- `tests_chronos_cloud.log`：运行 11 项，跳过 0 项，耗时 0.064 秒，失败标记=False
- `tests_agent_cloud.log`：运行 65 项，跳过 0 项，耗时 3.451 秒，失败标记=False
- `online_loop_cloud.log`：运行 29 项，跳过 1 项，耗时 10.364 秒，失败标记=False
- `tests_ett_cloud.log`：运行 5 项，跳过 0 项，耗时 1.335 秒，失败标记=False
- `tests_dual_cloud.log`：运行 18 项，跳过 0 项，耗时 0.032 秒，失败标记=False


汇总：144 项通过，1 项跳过，0 项失败。跳过项是 CUDA 跨设备一致性测试，因为该云端容器无 CUDA 设备。

### ETTh1 四臂受控概念漂移实验
- 每组 108 个完整反馈窗口；总计 432 个窗口。
- 仅在数据副本的 OT 通道注入漂移，原始 ETTh1.csv 未修改。
- 详细预测、反馈、监控、重训事件、版本事件和 ReAct 轨迹均保存在 `experiments/`。

| 策略 | Prequential MAE | MSE | 最差10窗口MAE | 触发 | 激活 |
|---|---:|---:|---:|---:|---:|
| Frozen | 1.670403 | 5.366424 | 2.782199 | 0 | 0 |
| Periodic | 1.638660 | 5.222721 | 2.731238 | 3 | 3 |
| Feedback ReAct | 1.650223 | 5.278552 | 2.741295 | 8 | 1 |
| Feedback No-Gate | 1.640981 | 5.206774 | 2.741295 | 3 | 3 |

结论：在本次固定数据切分和受控漂移强度下，Periodic 的整体 MAE 最低；ReAct 相比 Frozen 有改善，但未超过 Periodic。不能据此宣称反馈触发策略全面优胜。

### ATREA 触发流程演示
执行 `python -m atrea.cli --synthetic-demo` 成功。此演示使用合成反馈、合成特征和伪论文元数据，只验证触发与报告链路，不属于真实数据模型精度实验。

## 未完成 / 环境受限项目

1. 完整 NAS 搜索（8 个种群、2 代、5 epoch、3 个随机种子）在本云端 CPU 环境下超过单次执行时间限制，未生成可用的完整搜索结果。没有将不完整输出作为实验结论。
2. NAS 真实 ETTh1 训练适配器的 1 epoch smoke 已由 `tests_agent` 中的真实数据测试覆盖并通过；这不等于完整 NAS 搜索。
3. Chronos-2 的 11 项接口、门禁和切分测试通过，但没有执行真实 1000-step LoRA 微调。该云端容器没有 CUDA、没有预载模型权重，且当前运行未完成下载模型权重的步骤。
4. RTX 5060 GPU 实验不能在本云端 CPU 容器中复现。若需完整 GPU / Chronos-2 / 全量 NAS 结果，需要有 GPU 的运行时、模型权重与相应依赖。

## 产物索引
- `experiments/summary_table.csv`、`summary_table.md`：四臂汇总
- `experiments/metrics_detail.json`：详细指标
- `experiments/fig_*.png`：实验图
- `experiments/*/`：各组完整 JSONL 轨迹与版本事件
- `experiments/conclusion.md`：实验结论
- `cloud_run_artifacts/logs/`：测试与演示日志
