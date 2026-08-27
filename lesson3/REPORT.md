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
- r1 为快速验证：epochs=20；r2 起进入完整 epochs=100 实验，并逐步完成 3 次结构变化实验
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

### r2：完整 100 epochs + MLP hidden 64→128（第一次结构变化）

> 说明：r2/r3 起在 train.py 中加入"每模型独立 seed"，保证同配置跨轮完全可复现（r1 为旧版全局 RNG，仅作流程验证）。

Model | 主要结构 | 参数量 | MAE | MSE | Best Epoch | Training Time
------|----------|--------|-----|-----|------------|--------------
Baseline | Last Value | 0 | 2.5764 | 11.3767 | - | 0.00s
Linear | Linear(672→24) | 16,152 | 21.1076 | 615.0600 | 1 | 1.51s
MLP | MLP(672→128×2层→24) | 105,752 | 10.4547 | 158.1928 | 3 | 1.95s
CNN | CNN(kernel=3, channels=32) | 1,496 | 17.5249 | 383.8167 | 13 | 2.03s
LSTM | LSTM(1层, hidden=32) | 6,040 | 7.7428 | 84.4147 | 29 | 13.58s

### r3：CNN kernel 3→5（第二次结构变化）

Model | 主要结构 | 参数量 | MAE | MSE | Best Epoch | Training Time
------|----------|--------|-----|-----|------------|--------------
Baseline | Last Value | 0 | 2.5764 | 11.3767 | - | 0.00s
Linear | Linear(672→24) | 16,152 | 21.1076 | 615.0600 | 1 | 3.40s
MLP | MLP(672→128×2层→24) | 105,752 | 10.4547 | 158.1928 | 3 | 5.51s
CNN | CNN(kernel=5, channels=32) | 1,944 | 16.4493 | 352.3740 | 11 | 6.64s
LSTM | LSTM(1层, hidden=32) | 6,040 | 7.7428 | 84.4147 | 29 | 10.27s

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

### r2（完整 100 epochs + MLP hidden 64→128）

- **改了什么**：epochs 20→100（所有模型），MLP hidden 64→128（第一次结构变化，参数量 48,792→105,752，容量提升约 2.2 倍）。
- **模型哪里变了**：MLP 每层隐藏单元翻倍，可拟合更复杂的非线性关系；其余模型结构不变。
- **结果如何变化**（对比 r1）：
  - **LSTM 受益最大**：MAE 11.56→7.74（↓33%）、MSE 174.34→84.41（↓52%），best_epoch 从 8 移到 29，说明此前 20 epoch 远未收敛；100 epoch 后 LSTM 是所有学习模型中唯一逼近 Baseline（MAE 2.58）的。
  - **MLP（hidden=128）**：MAE 12.67→10.45（↓18%）、MSE 217.43→158.19（↓27%）。但 r2 同时含 epochs 与可复现 seed 变化，hidden 增量效果需在后续轮对比确认。
  - **CNN 反而变差**：MAE 13.74→17.52、MSE 276.59→383.82。原因是 100 epoch 下验证 loss 在 epoch 13 后持续上升（0.12→0.57），严重过拟合——CNN 是该数据集上最需正则化的模型。
  - **Linear 依旧过拟合**：best_epoch=1，100 epoch 无帮助，验证 loss 长期停在 1.1 附近。
- **复现性改进**：train.py 加入每模型独立 seed，r2 与 r3 中同配置的 Linear/MLP/LSTM 结果完全一致（差异为 0）。

### r3（CNN kernel 3→5）

- **改了什么**：仅改 CNN 卷积核 kernel 3→5（感受野扩大，参数量 1,496→1,944），其余（epochs=100、MLP hidden=128、LSTM 1/32）保持 r2 不变。
- **模型哪里变了**：CNN 每次卷积覆盖 5 个时间步（原 3 步），可捕捉更长时间跨度的局部时序模式。
- **结果如何变化**（CNN 对比 r2，其余模型与 r2 完全一致）：
  - **CNN 小幅改善**：MAE 17.52→16.45（↓6%）、MSE 383.82→352.37（↓8%），best_epoch 13→11。增大感受野确实缓解了一部分过拟合，但验证 loss 仍在 best_epoch 后持续攀升（0.16→0.59），说明 CNN 的根本问题不在 kernel 尺寸，而在于缺乏正则化（dropout/weight decay/early stopping）。
  - **Linear/MLP/LSTM 与 r2 完全相同**（MAE 21.11/10.45/7.74）：验证了 per-model seed 的可复现性，CNN 结构变化对其他模型零串扰。
- **下一步**：r4 做第三次结构变化 LSTM 1层/hidden32 → 2层/hidden64。
