# 实验结论

本实验在 ETTh1 加载后的数组副本中，从在线回放第 60 个 Episode 起对 OT 注入 +2℃ 均值偏移，并将偏差幅度放大至 1.5 倍；原始 CSV 未修改。四臂共享相同时间切分、数据和 seed=42，共得到 108 个完整反馈 Episode。整体 prequential MAE 最低的是 Periodic Retrain（1.6387），其次为 No-Gate（1.6410）、Feedback-Triggered ReAct（1.6502）和 Frozen（1.6704）。因此，本次实验中 ReAct 优于 Frozen，但没有超过 Periodic；最差 10-Episode MAE 也显示 Periodic 略好于 ReAct，结果不支持“反馈触发一定优于定期重训”。

Gate 对比中，Arm 3 MAE 为 1.6502，Arm 4 为 1.6410，启用 Gate 的整体 MAE 高约 0.0092℃，本次没有显示整体精度优势。ReAct 共触发 8 次候选工作流，7 次因近期改善不足被拒绝，1 次通过近期与冻结历史双条件后上线；这说明 Gate 有实际约束作用，但也可能拒绝部分有用更新，是否必要需结合安全性、最差阶段表现和其他漂移强度评估。恢复速度按 10-Episode 滑动 MAE 从 baseline×1.2 以上回到 baseline×1.1 以内定义：ReAct 为 12 个 Episode，Frozen 为 15 个，Periodic 与 No-Gate 均为 11 个。因此 ReAct 比 Frozen 快 3 个 Episode，却比 Periodic 和 No-Gate 慢 1 个。本轮启用了基于同一批成熟窗口反事实比较的回滚判定，但没有父版本满足回滚改善阈值，因此回滚计数为零；这不构成回滚机制有效性的证据。结果仅适用于当前受控漂移和固定切分，不代表 ETTh1 原始数据存在自然漂移；建议后续对多个漂移幅度和预先固定的随机种子复验。
