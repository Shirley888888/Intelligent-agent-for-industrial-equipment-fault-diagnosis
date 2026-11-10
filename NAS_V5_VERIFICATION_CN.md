# ATREA NAS V5 验证记录

验证日期：2026-11-08

## 自动化测试

- `python -m pytest -q`
- 结果：126 passed, 1 skipped, 1 warning, 11 subtests passed。
- 跳过项：CUDA 跨设备相关测试，当前执行环境无可用 CUDA。
- 警告：PyTorch Transformer nested tensor 性能提示，不影响测试通过。

## ETTh1 真实数据算子冒烟训练

- 数据：项目随附 ETTh1.csv。
- 输入窗口：96；预测长度：24。
- 训练轮数：1；batch size：512；随机种子：11；CPU。
- 算子：`amplitude_phase`。
- 状态：completed。
- 验证 MAE：0.7033076471（训练数据标准化目标尺度）。
- 验证 RMSE：0.8468430991（训练数据标准化目标尺度）。
- 参数量：62,233。
- 本次训练耗时：约 8.68 秒。
- split hash：`b3d897f6888a4eafe307b9077194bc0e3593cc26a6827df70842bc740403103b`。

这是算子能够接入真实 ETTh1 训练并完成反向传播/验证的冒烟测试，不是与原模型同预算、多随机种子、独立测试集比较的性能结论。

## NAS 调度与提案注入契约测试

使用测试回调验证了：输入弱点 `periodic phase drift` 后，搜索候选会携带 `operator_innovation` 研究提案元数据；多目标 Pareto 结果正常返回；`promotion_allowed` 保持 False。此项为调度契约测试，不是模型性能实验。

## 结论与限制

V5 已验证新算子前向、梯度、ETTh1 单轮真实训练、提案注入和 Pareto 选择逻辑。尚未证明新算子优于 LSTM/TCN/Transformer 或此前最佳 ATREA 模型。正式结论需要统一预算、多随机种子、完整训练、未触碰测试集及基线/消融对照。
