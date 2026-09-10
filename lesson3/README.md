# lesson3 · ETTh1 油温预测训练（从 Agent 到真实模型训练）

课程《工业智能体·第二课：从 Agent 到真实模型训练》课后实验任务 1 的落地项目。

- 数据：**真实 ETTh1 数据集**（电力变压器油温，7 变量：HUFL/HULL/MUFL/MULL/LUFL/LULL + OT，17,420 行，2016-07 至 2018-06）
- 任务：用过去 `seq_len` 小时的全部变量，预测未来 `pred_len` 小时的变压器油温 OT
- 模型：Baseline（Last Value）/ Linear / MLP / CNN / LSTM，训练后对比参数量、MAE、MSE、训练时间

## 运行方式

```bash
python train.py                       # 默认读取 config.yaml，自动取下一轮编号 r8, r9, ...
python train.py --round r8            # 指定轮次编号
python train.py --epochs 2            # 临时覆盖训练轮数（快速验证流程）
python evaluate.py --model linear --model lstm   # 最终方案确定后在 Test 上正式评价（一次）
python summary.py                     # 汇总 Validation 结果表 + 最终 Test 正式评价
```

## 流程修正（课前评审意见，v2）

1. **数据切分顺序**：先按时间把原始序列切成 Train/Val/Test（70/15/15），再在各段内部构造滑窗，
   段间样本互不跨越（消除旧版“先滑窗后切分”造成的边界样本重叠）。
2. **模型选择**：训练阶段仅在 Validation 上评估（`metrics.csv` 记录 `val_mae/val_mse`）；
   Test 只在最终方案确定后由 `evaluate.py` 做一次正式评价（`outputs/final_test_results.csv`）。
3. **CNN**：卷积后补充 ReLU 激活。
4. **Baseline 预测图**：反归一化到原始油温单位后再绘图。
5. **字段校验**：`utils.py` 严格校验 7 标准列齐全、非空、无 NaN/Inf，异常直接报错。

> 旧切分方式（先滑窗后切分）的 r1–r7 结果已归档为 `outputs/metrics_legacy_oldsplit.csv`，
> 不作为最终结论依据。
>
> **新切分方式已统一重跑完成（r8，2026-09-01）**：先按时间切原始序列 70/15/15、再段内滑窗，
> 5 个模型各 200 epochs 全量训练。当前结论见 `REPORT.md` 第 6 节：
> 最终方案为 **Linear**（Test MAE=1.3788），优于 Last-Value Baseline（Test MAE=1.4597）约 5.5%。

## 每轮调参工作流

1. 编辑 `config.yaml`（唯一调参入口）
   - 结构超参：`models.mlp.hidden / layers`、`models.cnn.kernel / channels`、`models.lstm.layers / hidden`
   - 训练超参：`training.epochs / batch_size / lr / seed`
   - 数据超参：`data.seq_len / pred_len / train_frac / val_frac`
2. 运行 `python train.py`（训练阶段只看 Validation 指标选型）
3. 产出：`outputs/metrics.csv`（逐轮累积，含 `val_mae/val_mse`）、`outputs/figures/*_loss_curve.png`（train/val loss 曲线）、`*_pred_vs_true.png`（Validation 预测 vs 真实）、`*_all_models_val_loss.png`（5 模型 val loss 对比）
4. 结构/超参定稿后：`python evaluate.py --model <最终模型>` → Test 正式评价
5. 每轮完成后更新本文件与 `REPORT.md` 并 push 到 GitHub

## 重大勘误（全量重跑）

**此前版本误用合成数据**：ETTh1.csv 曾由 `generate_synthetic_etth1.py` 生成（1500 行随机游走，
列名 f1..f6），旧下载脚本的 GitHub 路径已 404 失效。现已：
1. 替换为**真实 ETTh1 数据**（标准列名 HUFL..LULL+OT，17,420 行，2016-07 至 2018-06）；
2. 清空全部旧指标/图（`outputs/`），用真实数据重跑 r1–r7 全部实验序列；
3. 删除过时的 `extracted_project`、`run_results` 等错误阶段产物；修复 `check_imports.py` 的 BOM；
4. 修复 `train.py` best-权重加载 bug（`state_dict()` 需深拷贝，否则被后续训练 in-place 污染）。

## 调参历史（r1–r7 为旧切分方式存档）

| 轮次 | 改动内容 | 关键结果（真实 ETTh1） | 说明 |
|------|----------|------------------------|------|
| r1   | epochs=20 基准配置（快速验证） | LSTM(1/32) 最优：MAE=1.38、MSE=3.44（首次超越 Baseline MAE=1.46）；Linear=1.48；MLP=2.13；CNN=2.25 | CPU 训练；真实数据下模型表现与公开基准量级一致，见 REPORT.md |
| r2   | epochs 20→100 + MLP hidden 64→128 | MLP 1.84（小改善但依旧过拟合）；LSTM 保持最优 1.38 | 第一次结构变化，见 REPORT.md |
| r3   | CNN kernel 3→5 | CNN 2.26（几乎无变化）；LSTM 保持最优 1.38 | 第二次结构变化，见 REPORT.md |
| r4   | LSTM 1层/32 → 2层/64 | **LSTM(2/64) 全实验最优：MAE=1.366、MSE=3.56** | 第三次结构变化，见 REPORT.md |
| r5   | lr 0.001→0.0001（训练超参补测 1/3） | LSTM 1.38（略变差）；CNN 2.25（略好） | 见 REPORT.md |
| r6   | batch_size 256→128（训练超参补测 2/3） | **Linear 1.39（明显改善）**；LSTM 1.45（变差） | 见 REPORT.md |
| r7   | epochs 100→200（训练超参补测 3/3） | **Linear 1.376（追平 LSTM，零过拟合，仅 36s）**；LSTM 无改善 | 见 REPORT.md |
| r8   | **新切分正式重跑**：先切分后滑窗，全模型 200 epochs（lr=1e-4, bs=128） | **Linear 最终方案 Test MAE=1.3788**（优于 Baseline 1.4597 约 5.5%）；LSTM 新切分下 val 1.95 不再占优 | 结论依据，见 REPORT.md 第 4/5/6 节 |

## 环境

- Python 3.12 + PyTorch（GPU 优先，自动回落 CPU）
- 依赖见 `requirements.txt`

## 输出目录

```
outputs/
├── metrics.csv                          # 每轮全部模型 Validation 指标（累积，val_mae/val_mse）
├── metrics_legacy_oldsplit.csv          # 旧切分方式 r1–r7 结果存档（不作结论依据）
├── final_test_results.csv               # 最终方案 Test 正式评价（evaluate.py 生成）
├── unified_test_table.csv / .md         # summary.py 生成的模型选择统一表（Validation）
├── final_test_table.md                  # 最终 Test 正式评价表（summary.py 生成）
├── .round                               # 当前轮次计数器（自动维护）
└── figures/
    ├── rX_{model}_loss_curve.png        # 每模型 train/val loss 随 epoch 曲线
    ├── rX_{model}_pred_vs_true.png      # Validation 首个样本预测 vs 真实（反归一化）
    ├── rX_all_models_val_loss.png       # 5 模型 val loss 对比
    └── final_{model}_pred_vs_true.png   # Test 正式评价预测 vs 真实（evaluate.py 生成）
```
