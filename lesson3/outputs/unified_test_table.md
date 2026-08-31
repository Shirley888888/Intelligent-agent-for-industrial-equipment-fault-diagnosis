## 统一 Validation MAE 结果：各模型最佳表现

> 数据来源：Validation MAE（新切分方式：先按时间切分原始序列，再各段内滑窗）

| model    | best_round   | best_structure                          |   best_val_mae |   best_val_mse |   params | best_epoch   |   train_time_s |
|:---------|:-------------|:----------------------------------------|---------------:|---------------:|---------:|:-------------|---------------:|
| baseline | r8           | Last Value（取输入最后1个OT值重复24步） |         1.2839 |         2.9112 |        0 | -            |           0.02 |
| linear   | r8           | Linear(672→24)                          |         1.2983 |         2.8719 |    16152 | 200          |          58.45 |
| mlp      | r8           | MLP(672→128×2层→24)                     |         2.9186 |        12.7313 |   105752 | 25           |         112.96 |
| cnn      | r8           | CNN(kernel=5, channels=32)              |         2.4663 |         9.5671 |     1944 | 19           |          89.67 |
| lstm     | r8           | LSTM(2层, hidden=64)                    |         1.9498 |         6.0856 |    53528 | 36           |        1929.43 |

## 各模型 × 轮次 Validation MAE 明细

| model    |     r8 |
|:---------|-------:|
| baseline | 1.2839 |
| linear   | 1.2983 |
| mlp      | 2.9186 |
| cnn      | 2.4663 |
| lstm     | 1.9498 |

## 各模型 × 轮次 MSE 明细

| model    |      r8 |
|:---------|--------:|
| baseline |  2.9112 |
| linear   |  2.8719 |
| mlp      | 12.7313 |
| cnn      |  9.5671 |
| lstm     |  6.0856 |

---

## 最终方案 Test 正式评价（每次最终方案确定后仅评价一次）

> 以下 Test MAE/MSE 由 evaluate.py 产生，不用于任何调参，作为最终方案的正式证据。

| round   | model    | structure                               |   params |   test_mae |   test_mse |   train_time_s | best_epoch   | best_val_loss   | split_mode   |
|:--------|:---------|:----------------------------------------|---------:|-----------:|-----------:|---------------:|:-------------|:----------------|:-------------|
| final   | baseline | Last Value（取输入最后1个OT值重复24步） |        0 |     1.4597 |     3.9941 |           0.02 | -            | -               | time-first   |
| final   | linear   | Linear(672→24)                          |    16152 |     1.3788 |     3.5725 |          47.75 | 200          | 0.041205        | time-first   |