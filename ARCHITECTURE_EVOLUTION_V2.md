# ATREA 自主模型架构进化 V2

本次在 V1 基础上增量增加可执行的架构候选生成能力，原有 Linear/CNN1D/LSTM/TCN 名称和实现保留。

## 新增架构

- `LSTM_Attention`：LSTM 编码后增加多头自注意力与残差归一化。
- `RevIN_LSTM`：按样本对输入各变量进行可逆实例归一化，再对 OT 通道反归一化。
- `PatchTransformerLite`：将多变量时间窗切成 patch，使用 Transformer Encoder 建模。
- `DecompLinear`：对 OT 序列进行移动平均趋势/季节残差分解，使用双线性头预测。

## Agent 如何使用

`atrea.research.build_research_hypothesis()` 现在会把论文方法线索和残差诊断转成 `plan.candidate_architectures`。这些是结构化、可执行的模型规格，不再只是方法标签。宿主 `train_and_evaluate(proposal, batch_id)` 可直接读取该字段，按统一的时间切分、预处理和随机种子训练候选模型。

`atrea.architecture.run_architecture_sweep()` 提供统一候选执行适配器。宿主只需提供真实训练回调 `train_evaluate(spec, seed)`，回调必须返回真实指标。每个候选/种子独立记录，单个候选失败不会中断整个 sweep。

## 安全边界

- Agent 只能从注册架构目录中选择结构，并在白名单超参数内修改。
- 不执行 LLM 生成的任意 Python 代码。
- 对结构配置、注意力维度、通道数和参数量进行预算校验。
- 训练结果必须来自宿主真实训练回调；本模块不会伪造指标。
- 架构候选不自动上线。仍须经过现有冻结测试、基线比较、泄漏检查、回滚就绪检查和版本管理。

## 本轮能力边界

本轮实现了“注册组件组合式架构搜索”，可自动生成并执行多种新候选网络；尚未实现任意 Python 代码自主生成/热加载，也未声称已在用户本机 RTX 5060 上完成 GPU 训练。GPU 训练由用户本机环境中的宿主训练回调执行。

## 测试

运行：

```bash
python -m unittest tests_agent.test_architecture_evolution tests_agent.test_autonomous_research_engine -v
```
