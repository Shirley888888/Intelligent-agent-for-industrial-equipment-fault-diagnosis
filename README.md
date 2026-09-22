# ETTh1 Industrial ReAct Agent — LSTM Deployment

<!-- modify:model-unification -->
本项目的 **Agent 部署链路只使用 LSTM**。`agent/models/config.yaml` 中 `model.type: LSTM` 是唯一部署模型配置，`agent/models/best_model.pt` 必须与之匹配。历史训练/对比产物不参与 Agent 运行。

## Agent 全链路

<!-- modify:full-react-case -->
每个案例完整执行：**Forecast(LSTM) → Risk(训练集分位数) → RAG(Chunk + TF-IDF Embedding + Top-K + 相似度阈值) → 多步 ReAct Observation → Diagnosis → Final Answer**。原有 ETTh1 读取方式、异常检测和诊断规则保持不变。

## 动态风险阈值

<!-- modify:dynamic-risk -->
固定温度阈值已从 Agent 风险逻辑中移除。启动时只读取 ETTh1 的时间顺序训练段，并对 24h 窗口的温度峰值、温升、最大正斜率计算分位数阈值。所有分位数都在 `agent/models/config.yaml -> risk.quantiles` 中配置。当前演示验证后温度预警分位数设为 0.82；温升/斜率预警分位数为 0.75，高风险分位数为 0.92/0.95。程序启动会打印实际统计阈值。

## RAG / ReAct 配置

<!-- modify:rag-config -->
`rag.top_k`、`rag.similarity_threshold`、`rag.chunk_size`、`rag.chunk_overlap` 和 `react.max_steps` 全部位于配置文件，运行代码不新增检索超参魔法数字。每条检索证据保存 `id / score / accepted / text`，每次工具调用保存完整 Observation。

## 一次运行三个案例

```bat
python tests\run_agent_cases.py
```

或直接：

```bat
python run_agent.py
```

<!-- modify:case-persistence -->
程序固定按 `stable -> slow_rise -> fast_rise` 运行，并生成：

- `output/case_stable.json`
- `output/case_slow_rise.json`
- `output/case_fast_rise.json`
- `output/all_case_results.json`

每个 JSON 包含案例名称、96×7 原始输入、24h 预测、温度/温升/斜率指标、风险中间值和动态阈值、RAG 证据、全部 ReAct Observation、风险等级、诊断信息和 Agent 最终文本。

当前随项目数据和权重的 CPU 验证结果：`stable=LOW`、`slow_rise=MEDIUM`、`fast_rise=HIGH`。

## 单案例

```bat
python run_agent.py --case stable
python run_agent.py --case slow_rise
python run_agent.py --case fast_rise
```

## 环境

```bat
pip install -r requirements.txt
```

GPU 可用时默认使用 `cuda:0`；否则自动回退 CPU。数据集继续使用 `data/ETTh1.csv`。

## 四模型训练与单模型部署

<!-- modify:four-model-training -->
训练阶段现在比较 **Linear、CNN1D、LSTM、TCN** 四个候选模型。选择它们的目的分别是：Linear 提供强线性基线，CNN1D 捕捉局部时间模式，LSTM 建模递归长期依赖，TCN 使用扩张卷积获得较长感受野并支持并行训练。

四个模型使用同一 ETTh1 数据划分、96h 输入、24h OT 预测、同一指标体系与 early stopping，训练配置统一放在 `configs/experiment.yaml`。完整训练命令：

```bash
python train_gpu.py --device cuda:0
```

训练后 `outputs/metrics.csv`、`outputs/metrics.xlsx` 与 `outputs/metrics.json` 会给出 Persistence + 四模型对比结果，各模型最佳权重写入 `outputs/checkpoints/<Model>_best.pt`。**Agent 推理仍保持单模型部署**：当前 `agent/models/config.yaml` 继续只加载 LSTM，避免 ReAct 运行时出现多模型混杂。完成正式对比后，应依据验证集选择最终模型，再单独切换部署 checkpoint。
