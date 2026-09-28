# ETTh1 工业油温 Agent：实验收口版

本包在原项目基础上迭代，保留原模型、工具类和多步骤工作流。
唯一正式实验：`ETTh1_CLOSEOUT_20260929_v1`。来源：`results/final/experiment.json`。旧输出仅在`archive/pre_closeout/`追溯，禁止混入论文。

## 运行
在项目根目录打开终端。已有GPU环境先保留原CUDA版PyTorch，再安装其余依赖；`requirements-reproduced.txt`记录本次CPU评价环境，并非要求替换你现有的GPU环境。

```bash
python -m pip install -r requirements.txt
python run_agent.py --device cuda:0
python tests/evaluate_rag.py
python tests/test_risk_boundaries.py
python tests/test_diagnosis.py
python tests/test_closeout.py
```

数据、四个预测模型、Agent的LSTM权重、句向量模型以及冻结风险文件均已打包。无须重新训练或下载Embedding模型。`prepare_embedding.py`仅在模型文件缺失时用于联网恢复。
`run_agent.py`默认按stable、slow_rise、fast_rise顺序运行并保存到output/。这些文件属于复运行输出；论文仍以冻结的experiment.json为准。

## 完整复核
```bash
python calibrate_risk.py
python evaluate_final.py
python tests/evaluate_rag.py
python run_agent.py --device cpu
python tests/test_closeout.py
python render_results.py
```
`calibrate_risk.py`已有冻结结果时校验后复用，不根据案例调参。`render_results.py`只从正式源重新生成Markdown、CSV和PNG，不把临时复跑结果自动提升为新正式实验。Excel是同一源的冻结导出副本，非另一次实验。原GPU训练入口保留，新训练输出到独立training_runs目录。

## 本轮审计与优化

详见 `reports/AUDIT_AND_ANALYSIS.md` 和 `reports/REVISION_CHANGELOG.md`。
普通复跑的预测评价、RAG记录写入 `results/reruns/` 独立目录，不覆盖正式结果。
原正式实验对应的旧源码存于 `archive/closeout_v1_source/`，映射见 `results/final/source_snapshot.json`；当前修补源码及实测记录见 `results/audit/`。
运行 `python tests/test_revision.py` 检查修补行为，运行 `python analyze_results.py` 重建详细分析。
新上传包中的 `outputs/training_runs/20260929_035240/` 仅为训练运行记录，不作为第二套正式实验。

## 当前正式结果
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

## 阅读入口
- `MODIFICATION_SUMMARY.md`：分轮修改清单、文件位置和验证结果。
- `paper/PAPER_DRAFT.md`：中文论文框架及方法、实验粗稿。
- `paper/V8_PAPER_RESULTS.md`：正式数字。
- `paper/tables/Final_Experiment.xlsx`：统一Excel表。
- `results/final/rag_top3_records.csv`：两后端共72条候选记录；JSON保留完整文本。
- `configs/risk_thresholds.json`：训练校准、验证统计、数据校验值。

## 解释边界
知识库为25条英文人工演示知识，检索模型为预训练[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)。阈值预设，未用测试题调参。Embedding仍有无答案误接收，不能把相似候选当作有效答案保证。案例来自训练时段，不作为独立泛化证据。risk与历史anomaly/diagnosis分别输出。ReAct仅按多步骤工作流描述。当前不声称验证真实故障分类准确率。
