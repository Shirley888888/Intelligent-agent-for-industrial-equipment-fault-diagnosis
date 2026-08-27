# 实验报告：ETTh1 未来 24h 油温预测（课后实验·任务 1）

## 1. 实验目标

使用 ETTh1 数据，比较 Baseline / Linear / MLP / CNN / LSTM 模型在未来 24 小时 OT（变压器油温）预测上的表现，并完成至少 3 次结构变化实验：
- MLP hidden=64 → 128
- CNN kernel=3 → 5
- LSTM layers=1, hidden=32 → layers=2, hidden=64

## 2. 数据处理

- 输入：过去 96 小时，7 个变量（OT + f1/f2/f3/f4/f5/f6）
- 输出：未来 24 小时 OT
- 归一化：z-score，只用训练段拟合，避免泄漏
- 划分：严格按时间顺序 70/15/15（训练/验证/测试）
- 指标：MAE / MSE 均为还原到原始油温单位后计算

## 3. 实验设置

- 优化器 Adam，损失 MSELoss，lr=0.001，batch_size=256（可调，见 config.yaml）
- r1 为快速验证：epochs=20；后续轮次逐步完成 3 次结构变化实验
- 每轮全量重跑 5 个模型；loss 曲线见 `outputs/figures/`

## 4. 结果表

### r1：epochs=20 快速验证

Model | 主要结构 | 参数量 | MAE | MSE | Best Epoch | Training Time
------|----------|--------|-----|-----|------------|--------------
Baseline | Last Value | 0 | 2.5764 | 11.3767 | - | 0.00s
Linear | Linear(672→24) | 16,152 | 23.0944 | 699.3314 | 1 | 0.24s
MLP | MLP(672→64×2层→24) | 48,792 | 12.6669 | 217.4318 | 3 | 0.32s
CNN | CNN(kernel=3, channels=32) | 1,496 | 13.7382 | 276.5947 | 12 | 0.40s
LSTM | LSTM(1层, hidden=32) | 6,040 | 11.5613 | 174.3433 | 8 | 2.08s

> 逐轮结果详见 `outputs/metrics.csv`，曲线图见 `outputs/figures/`。

## 5. 结论与讨论

### r1（epochs=20 快速验证）

- **改了什么**：训练轮数设为 20，保持默认结构超参（MLP hidden=64/layers=2、CNN kernel=3/channels=32、LSTM 1层/hidden=32），完成 5 模型全流程。
- **模型哪里变了**：无结构变化；验证训练、验证、测试划分、z-score 归一化、逐 epoch 记录 loss、自动保存最优验证权重等流程是否正确。
- **结果如何变化/观察**：
  - **Baseline 依然强劲**：MAE 仅 2.58，说明该小数据集具有强自相关性，persistence 基线是很难超越的参照。
  - **Linear 严重过拟合**：训练 loss 降到 0.02，但验证 loss 在 epoch 1 后迅速上升并稳定在 1.1 左右，说明 672 维输入线性模型在 966 个训练样本上容量过大。
  - **MLP 最快收敛**：训练 loss 在 3 个 epoch 内接近 0，最佳验证 epoch=3，之后开始过拟合。
  - **LSTM 最稳定**：验证 loss 曲线平滑，在 epoch 8 达到最低，随后轻微反弹，综合 MAE/MSE 最优。
  - **CNN 中等**：验证 loss 在 epoch 12 最低，随后回升，结构与 kernel 尺寸仍需调优。
- **后续计划**：进入完整 epochs=100 基准，再依次完成 MLP hidden 64→128、CNN kernel 3→5、LSTM 1/32→2/64 三次结构变化实验。
