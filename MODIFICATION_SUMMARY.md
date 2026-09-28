# 本次修改清单

基于用户上传的 ETTh1_Industrial_Agent_V8_Paper_Demo2.zip 逐项迭代。未从零创建项目，原 models.py、data.py、metrics.py、预测/风险/检索/诊断工具架构和原训练主流程保留。修改处使用 `modify:` 简短注释。正式实验编号：ETTh1_CLOSEOUT_20260929_v1。

## 第一轮：RAG与风险规则

| 文件 | 具体修改 |
|---|---|
| agent/rag_tool.py | 原5条知识保留，新增20条，共25条；保留RAGTool、分块函数和retrieve接口；新增真正的SentenceTransformer句向量后端，使用归一化向量余弦相似度；TF-IDF作为独立基线保留；稳定排序；返回Top-3分数、文本、accepted。缺模型时明确报错，不冒充Embedding降级运行。 |
| agent/models/config.yaml | 检索默认Embedding，TF-IDF阈值0.08、Embedding阈值0.45预设；保留40词分块、8词重叠；三个风险指标均预设q90/q95，删除“按展示案例调参”的配置说明；新增冻结文件路径。 |
| prepare_embedding.py、agent/models/embedding/ | 固定预训练模型revision，提供恢复下载脚本；实际模型权重和分词配置随项目打包，正常运行无需联网。 |
| tests/rag_queries.json | 12条固定标注Query，直接、同义、多知识点、无答案各3条；多知识点标注全部目标知识ID。 |
| tests/evaluate_rag.py | 两后端分别运行12条Query；逐条保存Top-3、分数、是否过阈值、是否相关、拒答状态；统计Top1正确率、Hit@3、MRR@3、多证据覆盖及误拒/误收。 |
| agent/risk_tool.py | 保留原calculate_dynamic_thresholds/risk_level/analyze/summarize；取消缺失数据/列的静默替换和删除NaN后压缩时间；新增冻结文件读取及配置/数据哈希校验。 |
| calibrate_risk.py、configs/risk_thresholds.json | 仅从训练集计算固定分位数，验证集仅报告触发数；运行顺序在案例之前；已有冻结文件只复用或报配置不一致，不根据案例结果重标定。 |
| agent/agent.py | 将每次临时计算改为加载已冻结阈值；保留工具顺序、Observation日志及原有条件补检；说明其为多步骤工作流。 |
| tests/run_agent_cases.py | 保留依次执行和完整输出；删除强制LOW/MEDIUM/HIGH一一对应及“Adjust config quantiles”的异常提示，如实记录实际等级。 |

本轮固定后未依据12条检索测试题或三个展示结果调整任何阈值。

## 第二轮：正式实验统一与论文

| 文件/目录 | 具体修改 |
|---|---|
| agent/models/best_model.pt、scaler.pkl | 使用原包outputs/checkpoints/LSTM_best.pt及其scaler统计，消除部署模型与评价模型不一致。旧部署权重和scaler保留在历史归档。 |
| agent/predictor_tool.py | 保留原类和预测逻辑，补充scaler、features、target及输入输出长度与checkpoint的一致性校验。 |
| evaluate_final.py | 在同一时间划分上评价原四个checkpoint和Last Value，记录环境、哈希、原始预测和所有回归指标；没有重新训练或按测试分数挑选checkpoint。 |
| results/final/experiment.json | 唯一正式结果，聚合预测评价、风险冻结、案例输出、RAG明细、指标定义及来源。 |
| results/final/predictions.npz | 保存2494个测试窗口的真值和全部模型预测，支持从原始数组重算指标。 |
| results/final/rag_top3_records.csv | 两后端×12条Query×3候选，共72条完整检索审计记录。 |
| render_results.py | 统一从正式JSON生成README、论文结果、初稿中的表格、CSV和PNG，以及生成文件哈希；普通复跑不自动替换正式源。 |
| paper/tables/ | 正式模型、案例、RAG的CSV/Markdown；Final_Experiment.xlsx含3张统一表，来自同一JSON，已检查导出数值并渲染查看。 |
| paper/figures/ | 更新MAE/MSE、三个案例和RAG对比，共6张PNG。 |
| outputs/metrics.*、results/*comparison.csv、results/case_study_results.csv、output/ | 兼容路径统一使用正式源中的同一数值；案例JSON保存原输入、预测、风险、证据及工具日志。 |
| paper/PAPER_DRAFT.md | 建立题目、摘要、引言、方法、实验、结果、结论一级标题；方法与实验写粗稿，明确数据切分、分位数协议、RAG指标分母与实验局限。 |
| README.md、paper/V8_PAPER_RESULTS.md | 统一运行说明、模型身份、表格和结论；说明旧数值不可引用，ReAct不作为开放式推理创新点。 |
| train_gpu.py | 仅将后续训练产物写入outputs/training_runs/时间戳，防止覆盖已固定checkpoint和正式表格；训练算法不改。 |
| archive/pre_closeout/ | 原表图、TensorBoard记录、lesson3历史输出、旧说明和旧部署权重完整保留用于追溯；旧说明入口改为指向当前README与正式数据。 |
| requirements.txt、requirements-reproduced.txt | 新增句向量依赖，补充原训练Excel导出/进度条依赖；记录本次验证环境，不要求替换用户现有GPU版PyTorch。 |

## 第三轮：验证与打包

- tests/test_closeout.py：新增针对本次要求的有效验证，覆盖数据/配置/模型哈希、LSTM权重一致性、冻结阈值、原始预测重算、12条标注与指标重算、长文分块重叠、三个案例、CSV和文档图表哈希。
- 原tests/test_risk_boundaries.py及tests/test_diagnosis.py均运行通过，未改写其预期结果来迎合本次输出。
- 实际运行两个RAG后端及三个完整Agent案例，保存输出；验证日志见results/final/verification.log。
- 保留代码和原数据，历史结果单独归档；交付ZIP排除版本控制元数据、IDE缓存和Python字节码，这些不属于运行依赖。

## 如实保留的结果与限制

- TF-IDF：Top1正确率44.44%，Hit@3 77.78%，MRR@3 0.592593，无答案拒答0/3。
- Embedding：Top1正确率77.78%，Hit@3 88.89%，MRR@3 0.833333，无答案拒答2/3，仍有1条误接收。
- 三案例risk：stable=LOW、slow_rise=LOW、fast_rise=MEDIUM。没有反向调参制造三级分离。
- 三案例diagnosis：NORMAL、WARNING、ANOMALY。这来自保留的历史异常规则与预测风险组合，不能与risk混称。
- 三案例来自训练时段，仅作为冻结规则后的固定展示，不能称为独立未见测试。
- 正式模型评价为现有checkpoint的CPU复核，未执行本地RTX GPU训练。历史GPU结果存在小数差异时只引用当前正式源。
- 25条英文知识为人工演示内容；不声称经过工业专家审查，不声称12题可证明广泛语义能力；未新增多轮检索研究或其他无关功能。
