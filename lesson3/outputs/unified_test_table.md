## 统一 Validation MAE 结果：各模型最佳表现

> 数据来源：Validation MAE（新切分方式：先按时间切分原始序列，再各段内滑窗）

| model    | best_round   | best_structure                          |   best_val_mae |   best_val_mse |   params | best_epoch   |   train_time_s |
|:---------|:-------------|:----------------------------------------|---------------:|---------------:|---------:|:-------------|---------------:|
| baseline | r8           | Last Value（取输入最后1个OT值重复24步） |         1.2839 |         2.9112 |        0 | -            |           0.04 |
| linear   | r8           | Linear(672→24)                          |         2.8124 |        12.69   |    16152 | 2            |           0.4  |
| mlp      | r8           | MLP(672→128×2层→24)                     |         3.816  |        19.0451 |   105752 | 2            |           0.55 |
| cnn      | r8           | CNN(kernel=5, channels=32)              |        11.7158 |       143.83   |     1944 | 2            |           0.67 |
| lstm     | r8           | LSTM(2层, hidden=64)                    |         8.405  |        76.0264 |    53528 | 2            |          10.6  |

## 各模型 × 轮次 Validation MAE 明细

| model    |      r8 |
|:---------|--------:|
| baseline |  1.2839 |
| linear   |  2.8124 |
| mlp      |  3.816  |
| cnn      | 11.7158 |
| lstm     |  8.405  |

## 各模型 × 轮次 MSE 明细

| model    |       r8 |
|:---------|---------:|
| baseline |   2.9112 |
| linear   |  12.69   |
| mlp      |  19.0451 |
| cnn      | 143.83   |
| lstm     |  76.0264 |

---

## 最终方案 Test 正式评价（每次最终方案确定后仅评价一次）

> 以下 Test MAE/MSE 由 evaluate.py 产生，不用于任何调参，作为最终方案的正式证据。

| round   | model    | structure                               |   params |   test_mae |   test_mse |   train_time_s | best_epoch   | best_val_loss   | split_mode   |
|:--------|:---------|:----------------------------------------|---------:|-----------:|-----------:|---------------:|:-------------|:----------------|:-------------|
| final   | baseline | Last Value（取输入最后1个OT值重复24步） |        0 |     1.4597 |     3.9941 |           0.02 | -            | -               | time-first   |
| final   | linear   | Linear(672→24)                          |    16152 |     2.813  |    13.124  |           0.39 | 2            | 0.182074        | time-first   |