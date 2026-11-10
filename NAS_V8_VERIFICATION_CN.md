# NAS V8 验证记录

## 自动化测试

- 全量测试：`python -m unittest discover -s tests_agent -p 'test_*.py' -v`
- 结果：61 项测试全部通过。
- 新增覆盖：DARTS 架构 logits 可反向传播并更新；离散基因型导出；符号回归可恢复合成时间趋势；错误驱动创新候选包含 DARTS 节点。

## ETTh1 训练冒烟测试

使用工程内置 ETTh1，输入窗口 96、预测长度 24、CPU、batch size 256、1 epoch。测试候选包含 DARTS 混合节点（spectral_gain、depthwise_conv、trend_split、pointwise_mlp）与 LSTM。

| 指标 | 结果 |
|---|---:|
| MAE | 0.540510 |
| RMSE | 0.654548 |
| 训练 epoch | 1 |
| 参数量 | 14,142 |
| 训练耗时 | 5.46 s |
| 验证窗口数 | 1,731 |
| split hash | `b3d897f6888a4eafe307b9077194bc0e3593cc26a6827df70842bc740403103b` |

另行执行了架构离散化检查，当前这次随机初始化/单步更新导出的最高权重原语为 `trend_split`。这只是接口冒烟验证，不代表它是最优算子。

## 结论与限制

以上结果证明代码可通过测试、可在 ETTh1 上训练并可导出离散架构；不证明 V8 优于 V7 或任何基线。训练只有一个随机种子、一个 epoch，未进行多种子统计、完整训练预算和最终独立测试集评估。V8 的可微搜索是 DARTS-style 联合梯度混合操作，不是完整 bilevel DARTS。符号回归当前产出可解释研究结果，不会自动写入线上预测模型。
