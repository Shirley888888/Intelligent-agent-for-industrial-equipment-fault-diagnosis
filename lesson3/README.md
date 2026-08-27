# lesson3 · ETTh1 油温预测训练（从 Agent 到真实模型训练）

课程《工业智能体·第二课：从 Agent 到真实模型训练》课后实验任务 1 的落地项目。

- 数据：ETTh1（电力变压器油温数据集，7 变量：OT + f1/f2/f3/f4/f5/f6）
- 任务：用过去 `seq_len` 小时的全部变量，预测未来 `pred_len` 小时的变压器油温 OT
- 模型：Baseline（Last Value）/ Linear / MLP / CNN / LSTM，训练后对比参数量、MAE、MSE、训练时间

## 运行方式

```bash
cd lesson3
python train.py                 # 默认读取 config.yaml，自动取下一轮编号 r1, r2, ...
python train.py --round r3      # 指定轮次编号
```

## 每轮调参工作流

1. 编辑 `config.yaml`（唯一调参入口）
   - 结构超参：`models.mlp.hidden / layers`、`models.cnn.kernel / channels`、`models.lstm.layers / hidden`
   - 训练超参：`training.epochs / batch_size / lr / seed`
   - 数据超参：`data.seq_len / pred_len / train_frac / val_frac`
2. 运行 `python train.py`
3. 产出：`outputs/metrics.csv`（逐轮累积）、`outputs/figures/*_loss_curve.png`（train/val loss 曲线）、`*_pred_vs_true.png`（预测 vs 真实）、`*_all_models_val_loss.png`（5 模型 val loss 对比）
4. 每轮完成后更新本文件与 `REPORT.md` 并 push 到 GitHub

## 重要勘误（r7 后全量重跑）

发现并修复了 `train.py` 中 best-权重加载 bug：`model.state_dict()` 返回参数**引用**而非拷贝，
被后续训练 in-place 污染，导致测试评估误用了最后一个 epoch 的权重。
已改为深拷贝并重跑 r1–r7 全部轮次，REPORT.md / metrics.csv 均为修复后的数值
（val loss 曲线与 best_epoch 记录原本正确、不受影响）。

## 调参历史

| 轮次 | 改动内容 | 关键结果 | 说明 |
|------|----------|----------|------|
| r1   | epochs=20 基准配置（快速验证） | LSTM 最优：MAE=9.22、MSE=119.49；Baseline 次优：MAE=2.58 | CPU 训练；小数据集下 persistence 基线很强，见 REPORT.md |
| r2   | epochs 20→100 + MLP hidden 64→128（每模型独立 seed） | LSTM MAE 9.22→8.37（↓9%）；MLP best-epoch 过早导致 test 偏大（16.61→23.03） | 第一次结构变化；train.py 加入可复现 seed，同配置跨轮结果完全一致，见 REPORT.md |
| r3   | CNN kernel 3→5 | CNN MAE 15.86→15.20（↓4%）、MSE 344→336；其余模型与 r2 完全一致 | 第二次结构变化；增大感受野小幅改善 CNN 但仍过拟合，见 REPORT.md |
| r4   | LSTM 1层/32 → 2层/64 | LSTM MAE 8.37→5.46（↓35%）、MSE 98→41（↓58%）；其余模型与 r2/r3 完全一致 | 第三次结构变化；LSTM 为学习模型最优（全实验最佳 MAE 5.46）并逼近 Baseline，见 REPORT.md |
| r5   | lr 0.001→0.0001（训练超参补测 1/3） | CNN MAE 15.20→13.33（↓12%）、MLP 23.03→15.33（↓33%）、Linear 31.32→23.29；LSTM 反而变差 5.46→7.95 | 小 lr 改善 Linear/MLP/CNN 的 best-epoch 泛化；LSTM 暴露 val/test 时段分布偏移（val 最优但 test 最差），见 REPORT.md |
| r6   | batch_size 256→128（训练超参补测 2/3） | 各模型基本持平（LSTM 7.95→7.97、CNN 13.33→13.34、MLP 15.33→15.41）；Linear 略变差 23.29→24.01 | 本任务下 batch 减半无明显收益/损失，见 REPORT.md |
| r7   | epochs 100→200（训练超参补测 3/3） | 与 r6 完全一致（best-epoch 早已选定），仅 LSTM 耗时 47s→112s | 已有 early stopping 时多余 epochs 纯耗时无收益，见 REPORT.md |

## 环境

- Python 3.12 + PyTorch（GPU 优先，自动回落 CPU）
- 依赖见 `requirements.txt`

## 输出目录

```
outputs/
├── metrics.csv                          # 每轮全部模型指标（累积）
├── .round                               # 当前轮次计数器（自动维护）
└── figures/
    ├── rX_{model}_loss_curve.png        # 每模型 train/val loss 随 epoch 曲线
    ├── rX_{model}_pred_vs_true.png      # 测试集首个样本预测 vs 真实
    └── rX_all_models_val_loss.png       # 5 模型 val loss 对比
```
