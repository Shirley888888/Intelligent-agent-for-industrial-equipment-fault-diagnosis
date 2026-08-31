## 最终方案 Test 正式评价（每次最终方案确定后仅评价一次）

> 以下 Test MAE/MSE 由 evaluate.py 产生，不用于任何调参，作为最终方案的正式证据。

| round   | model    | structure                               |   params |   test_mae |   test_mse |   train_time_s | best_epoch   | best_val_loss   | split_mode   |
|:--------|:---------|:----------------------------------------|---------:|-----------:|-----------:|---------------:|:-------------|:----------------|:-------------|
| final   | baseline | Last Value（取输入最后1个OT值重复24步） |        0 |     1.4597 |     3.9941 |           0.02 | -            | -               | time-first   |
| final   | linear   | Linear(672→24)                          |    16152 |     2.813  |    13.124  |           0.39 | 2            | 0.182074        | time-first   |