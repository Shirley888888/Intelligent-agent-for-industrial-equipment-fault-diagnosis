# 本轮逐项修改清单

本轮编号：AUDIT_R2_20260929。基于本次上传Closeout(1)包迭代，原正式实验编号与模型权重不变。上一轮完整清单仍保留在根目录MODIFICATION_SUMMARY.md。

## 第一轮：审计与修补

|位置|具体变化|解决的问题|
|---|---|---|
|agent/rag_tool.py::__post_init__|按backend选择对应阈值，验证top_k、阈值有限性和范围|直接切换TF-IDF时误用Embedding阈值|
|agent/rag_tool.py::retrieve / retrieve_many / _rank_scores|保留单条接口，新增批量查询并共用排序函数，拒绝空白查询|减少重复编码调用，不改变检索政策|
|agent/__init__.py|使用延迟导入，保留OilTemperatureAgent公开接口|仅使用RAG或风险工具时被迫导入torch与完整Agent|
|experiment_paths.py|统一创建独立复跑目录，禁止写正式、论文、历史归档以及其父目录，禁止覆盖非空目录|正式结果被普通复跑覆盖|
|evaluate_final.py::main|新增可选output_dir，CLI新增--output-dir，默认写时间戳目录|不再覆盖正式预测数组与评价JSON|
|tests/evaluate_rag.py|批量编码固定Query，新增隔离输出目录|不再覆盖正式RAG记录|
|run_agent.py::load_history / main|添加main保护；统一CSV读取；CSV必须有date且恰好96行；拒绝重复、不连续、逆序时间；--input与--case互斥|模块不能安全导入、入口校验不一致、时间斜率可能被错误解释|
|tests/run_agent_cases.py|复用load_history|三案例与CLI采用相同输入规则|
|data.py::load_splits|校验小时序列、有限数、划分和窗口参数|训练数据不规则时静默生成窗口|
|agent/predictor_tool.py|先检查scaler形状、有限性和正比例，再比较checkpoint；核对目标索引|广播异常及无效缩放参数|
|agent/risk_tool.py::calculate_dynamic_thresholds|滑动窗口视图替代Python逐窗循环，保留指标和短序列行为|降低重复统计开销|
|agent/risk_tool.py::load_frozen_thresholds|根据训练数据重算并核对冻结分位数|仅检查配置和数据哈希而未验证冻结数值|
|calibrate_risk.py|校验风险/预测的训练比例及预测长度；验证比例读统一配置；已有冻结文件也校验实际数值|避免校准协议漂移|
|render_results.py|禁止assemble覆盖已有正式实验；渲染不覆盖实时output；更新README与论文说明|渲染可能掩盖复跑差异与正式来源混淆|
|archive/closeout_v1_source/、results/final/source_snapshot.json|保存原实验对应源码及映射|保留原来源哈希，不把新代码伪装成产生旧实验的代码|
|tests/test_closeout.py|原源码哈希经快照核对，模型、数据、阈值和结果仍严格核对|使历史实验和当前实现的校验职责明确|

本轮没有修改知识条目、12条Query、gold标注、q90/q95、TF-IDF/Embedding阈值数值；没有按案例反向调参；没有修改原模型类、训练损失或ReAct工具顺序。

## 第二轮：详细分析与优化验证

|位置|新增内容|
|---|---|
|analyze_results.py|从冻结预测与原始CSV生成逐步长误差、模型改善率、窗口胜率、峰值偏差、斜率MAE、检索类别与失败记录、风险分布和新上传训练权重比较|
|reports/AUDIT_AND_ANALYSIS.md|逐项完成度、定量分析、已完成优化、未被实验证明的事项和论文解释边界|
|reports/figures/horizon_mae.png|24小时逐步长MAE曲线|
|results/audit/*.csv、analysis.json|上述详细分析的可复核数据|
|tests/test_revision.py|阈值切换、无效查询、输出路径保护、时间轴、冻结值篡改、向量化等价、批量检索等价及五次中位耗时测试|
|tests/verify_reproduction.py|正式与真实复跑逐项比较回归指标、预测数组、全部RAG排名/阈值通过、实现哈希|
|results/audit/implementation_manifest.json|当前修订代码哈希，与历史正式实验来源分开记录|
|results/audit/verification.log|本轮执行命令、通过状态与输出日志|

风险等级分析采用原始CSV真实值，避免由float32标准化再反标准化的微小舍入误差在阈值等号处造成错误分类。该补充分析不改变既有正式回归表。

## 实际优化收益

- 相同输入下训练分位数新旧逐值一致；本机五次中位耗时约提升6.1倍。
- 12条句向量检索批量与单条的Top-3排序、accepted完全一致，浮点分数最大差约1.19e-7；本机五次中位耗时约提升2.9倍（不含载入）。
- 相同固定checkpoint复跑保持指标在1e-5绝对容差内，检索统计完全一致。
- 这些是计算和复现可靠性优化，不宣称预测精度或检索正确率提高。

## 尚存的研究局限

Q11无答案误接收仍保留；12题不足以说明工业泛化；测试时段缺少HIGH规则等级样本；原三案例属于训练期展示；未开展真实故障标签评价。未为“优化”而加入无关功能或伪造更好的实验结果。
