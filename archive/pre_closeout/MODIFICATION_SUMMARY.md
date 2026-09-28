# V8 三项要求逐项复核与查漏补缺

> 本轮以用户上传的 `ETTh1_Industrial_Agent_V8_Paper_Demo(3).zip` 为唯一基准进行复核；不重构 RAG / Chunk / Embedding / ReAct 核心业务逻辑。  <!-- modify:audit-base -->

## 任务1：stable / slow_rise / fast_rise 完整运行与 JSON 持久化
- `tests/run_agent_cases.py` 固定按 `stable -> slow_rise -> fast_rise` 执行，每例调用 `OilTemperatureAgent.run()`，经过 Forecast / Risk / RAG / ReAct Observation / Diagnosis / Final Answer。
- 每例输出到 `output/case_{case_name}.json`，汇总到 `output/all_case_results.json`。
- 每例字段包含：案例名、96×7 原始输入、24h 预测、温度/温升/斜率、风险中间值与阈值、RAG `id/score/accepted/text`、ReAct Observation 日志、风险等级、Agent 最终文本、模型、异常与诊断结果。
- 本轮重新实际执行后：`stable=LOW`、`slow_rise=MEDIUM`、`fast_rise=HIGH`。
- 查漏补缺：旧 `tests/cases/*/prediction.json` 和 `tests/cases/all_case_results.json` 是历史生成物且分位数元数据已过期，已删除，避免与当前 `output/` 结果混淆；`AGENT_README.md`、`tests/README.md` 已统一为真实输出路径。  <!-- modify:audit-stale-output -->

## 任务2：ETTh1 训练集动态分位数风险阈值
- `agent/risk_tool.py` 从 ETTh1 时间顺序训练段计算 24h 窗口的温度峰值、温升和最大正斜率分布。
- `agent/models/config.yaml` 是分位数参数唯一配置来源：temperature 0.82/0.92，rise 0.75/0.95，slope 0.75/0.95。
- Agent 启动时计算并打印实际阈值，再把 thresholds 显式传给 Risk 模块。
- 本轮全项目代码/配置检查未发现 75℃/90℃ 风险常量；仅测试文档保留“旧 75/90 已被替代”的说明文字。
- 本轮实测：temperature q0.82=25.816999 / q0.92=34.750999；rise q0.75=4.537499 / q0.95=7.176001；slope q0.75=2.532000 / q0.95=3.728001。

## 任务3：Agent 实际预测模型统一
- Agent 部署模型统一为 **LSTM**：`agent/models/config.yaml -> model.type: LSTM`。
- `agent/models/best_model.pt` 本轮实际读取确认 `model_name=LSTM`。
- `agent/predictor_tool.py` 只允许构造 LSTM，并校验 checkpoint 的 `model_name` 与 config 一致；不存在 TCN/Linear 部署加载分支。
- `README.md` / `AGENT_README.md` 的 Agent 部署说明统一为 LSTM。
- 根目录训练实验仍可比较 Linear/CNN1D/LSTM/TCN；它属于离线候选模型训练，不参与 Agent 推理。README 已明确区分“训练候选模型”和“Agent 单模型部署”，避免再次把实验模型误认为运行时模型。

## 本轮实际验证
- `python tests/run_agent_cases.py`：通过，LOW / MEDIUM / HIGH 三案例区分成立。
- `python tests/test_risk_boundaries.py`：全部通过。
- `python tests/test_diagnosis.py`：4/4 通过。
- `python -m compileall -q agent tests run_agent.py`：通过。
- checkpoint 元数据：`model_name=LSTM`, `best_epoch=5`。

## 运行命令
```bat
conda activate etth1_gpu
python tests\run_agent_cases.py
```

运行后查看：
- `output/case_stable.json`
- `output/case_slow_rise.json`
- `output/case_fast_rise.json`
- `output/all_case_results.json`
