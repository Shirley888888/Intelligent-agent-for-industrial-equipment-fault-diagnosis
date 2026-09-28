# 正式实验结果

数据源：`results/final/experiment.json`，实验编号 ETTh1_CLOSEOUT_20260929_v1。

| Model | MAE | MSE | PeakError |
| --- | --- | --- | --- |
| Persistence | 1.459667 | 3.994103 | 2.167372 |
| Linear | 1.619881 | 4.469734 | 1.593754 |
| CNN1D | 2.150202 | 7.759851 | 2.371695 |
| LSTM | 1.354184 | 3.476774 | 1.359466 |
| TCN | 1.557262 | 3.925312 | 1.578792 |

| Case | Model | Peak_C | Rise_C | Slope_C_per_h | Risk | Diagnosis |
| --- | --- | --- | --- | --- | --- | --- |
| stable | LSTM | 25.058632 | 0.889136 | 0.683664 | LOW | NORMAL |
| slow_rise | LSTM | 30.172935 | 0.000000 | 0.844649 | LOW | WARNING |
| fast_rise | LSTM | 34.051044 | 1.040500 | 0.769422 | MEDIUM | ANOMALY |

| Backend | retrieval_accuracy_top1_thresholded | Hit@3_raw | MRR@3_raw | no_answer_rejection_rate |
| --- | --- | --- | --- | --- |
| tfidf | 0.444444 | 0.777778 | 0.592593 | 0.000000 |
| embedding | 0.777778 | 0.888889 | 0.833333 | 0.666667 |
