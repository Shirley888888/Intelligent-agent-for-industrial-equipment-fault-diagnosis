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

## 调参历史

| 轮次 | 改动内容 | 关键结果 | 说明 |
|------|----------|----------|------|
| r1   | epochs=20 基准配置（快速验证） | LSTM 最优：MAE=11.56、MSE=174.34；Baseline 次优：MAE=2.58 | CPU 训练；小数据集下 persistence 基线很强，MLP/CNN/Linear 仍有优化空间，见 REPORT.md |
| r2   | epochs 20→100 + MLP hidden 64→128（每模型独立 seed） | LSTM MAE 11.56→7.74、MSE 174→84；MLP MAE 12.67→10.45；CNN 过拟合变差 13.74→17.52 | 第一次结构变化；train.py 加入可复现 seed，同配置跨轮结果完全一致，见 REPORT.md |
| r3   | CNN kernel 3→5 | CNN MAE 17.52→16.45、MSE 384→352；其余模型与 r2 完全一致 | 第二次结构变化；增大感受野小幅改善 CNN 但仍过拟合，见 REPORT.md |
| r4   | LSTM 1层/32 → 2层/64 | LSTM MAE 7.74→5.31（↓31%）、MSE 84→39（↓53%）；其余模型与 r2/r3 完全一致 | 第三次结构变化；LSTM 为学习模型最优并逼近 Baseline，三次结构变化实验完成，见 REPORT.md |
| r5   | lr 0.001→0.0001（训练超参补测 1/3） | CNN MAE 16.45→13.23（↓20%）、MSE 352→256；MLP MSE 158→122；LSTM 反而变差 5.31→6.52 | 小 lr 对 CNN/MLP/Linear 是正收益（CNN 从欠拟合转向充分训练，best_epoch 11→94），对 LSTM 收敛过慢变差，见 REPORT.md |
| r6   | batch_size 256→128（训练超参补测 2/3） | LSTM MAE 6.52→5.28（↓19%）、MSE 58→43；MLP MAE 9.53→8.75；CNN MAE 13.23→12.60；Linear 变差 20.34→23.70 | 小 batch 的梯度噪声等效隐式正则，对 MLP/CNN/LSTM 均正收益、对 Linear 负收益；小 lr+小 batch 组合 LSTM 最优，见 REPORT.md |

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
