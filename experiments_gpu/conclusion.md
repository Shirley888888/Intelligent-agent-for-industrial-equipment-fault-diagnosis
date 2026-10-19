# 实验结论

本实验在 ETTh1 加载后的副本上实施受控概念漂移，起点为行 16247，仅对 OT 通道施加均值偏移 2.0℃ 并将偏差幅度放大至 1.5 倍；原始 CSV 未修改。整体 prequential MAE 最低的是 Arm 4 - Feedback-Triggered No-Gate。ReAct 的 MAE 为 1.5336，Frozen 为 1.5822，Periodic 为 1.6107，因此应依据数值而非机制预期判断反馈触发策略是否优于固定模型和定期重训。

Gate 对比方面，Arm 3 MAE 为 1.5336，去 Gate 的 Arm 4 为 1.5294；差值（No-Gate 减 ReAct）为 -0.0041。正值表示启用 Gate 的误差较低，负值则表示本次回放中 Gate 未带来 MAE 优势；还需结合最差阶段 MAE 与上线次数解释。恢复速度按 10-Episode 滑动 MAE 从 baseline×1.2 以上回到 baseline×1.1 以内定义：Arm 3 为 2 个 Episode，Frozen 为 2 个 Episode，Periodic 为 4 个 Episode，No-Gate 为 15 个 Episode。若未恢复或未出现退化，不将其伪装为零。该结论仅适用于本次受控漂移强度和固定切分，不能直接外推为 ETTh1 原始数据存在自然概念漂移。
