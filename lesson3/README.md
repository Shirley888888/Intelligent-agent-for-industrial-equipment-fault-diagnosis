# lesson3 · ETTh1 油温预测训练（从 Agent 到真实模型训练）

课程《工业智能体·第二课：从 Agent 到真实模型训练》课后实验任务 1 的落地项目。

- 数据：**真实 ETTh1 数据集**（电力变压器油温，7 变量：HUFL/HULL/MUFL/MULL/LUFL/LULL + OT，17,420 行，2016-07 至 2018-06）
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

## 重大勘误（全量重跑）

**此前版本误用合成数据**：ETTh1.csv 曾由 `generate_synthetic_etth1.py` 生成（1500 行随机游走，
列名 f1..f6），旧下载脚本的 GitHub 路径已 404 失效。现已：
1. 替换为**真实 ETTh1 数据**（标准列名 HUFL..LULL+OT，17,420 行，2016-07 至 2018-06）；
2. 清空全部旧指标/图（`outputs/`），用真实数据重跑 r1–r7 全部实验序列；
3. 删除过时的 `extracted_project`、`run_results` 等错误阶段产物；修复 `check_imports.py` 的 BOM；
4. 修复 `train.py` best-权重加载 bug（`state_dict()` 需深拷贝，否则被后续训练 in-place 污染）。

## 调参历史

| 轮次 | 改动内容 | 关键结果（真实 ETTh1） | 说明 |
|------|----------|------------------------|------|
| r1   | epochs=20 基准配置（快速验证） | LSTM(1/32) 最优：MAE=1.38、MSE=3.44（首次超越 Baseline MAE=1.46）；Linear=1.48；MLP=2.13；CNN=2.25 | CPU 训练；真实数据下模型表现与公开基准量级一致，见 REPORT.md |
| r2   | epochs 20→100 + MLP hidden 64→128 | MLP 1.84（小改善但依旧过拟合）；LSTM 保持最优 1.38 | 第一次结构变化，见 REPORT.md |
| r3   | CNN kernel 3→5 | CNN 2.26（几乎无变化）；LSTM 保持最优 1.38 | 第二次结构变化，见 REPORT.md |
| r4   | LSTM 1层/32 → 2层/64 | **LSTM(2/64) 全实验最优：MAE=1.366、MSE=3.56** | 第三次结构变化，见 REPORT.md |
| r5   | lr 0.001→0.0001（训练超参补测 1/3） | LSTM 1.38（略变差）；CNN 2.25（略好） | 见 REPORT.md |
| r6   | batch_size 256→128（训练超参补测 2/3） | **Linear 1.39（明显改善）**；LSTM 1.45（变差） | 见 REPORT.md |
| r7   | epochs 100→200（训练超参补测 3/3） | **Linear 1.376（追平 LSTM，零过拟合，仅 36s）**；LSTM 无改善 | 见 REPORT.md |

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
