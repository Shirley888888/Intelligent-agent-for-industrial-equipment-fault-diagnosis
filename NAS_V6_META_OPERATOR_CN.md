# ATREA NAS V6：受控元算子发现（Meta-Operator Discovery）

## 本版目标

V5 的搜索空间主要由已注册网络模块组成。V6 增加“基础计算原语 → 有序计算图 → PyTorch 模块”的安全编译链，让 Agent 可以设计新的模块内部计算路径，而不只是排列 LSTM/TCN/Transformer 等现成模块。

## 新增能力

### 1. 元算子基础原语

当前白名单包含 8 种原语：

- `layer_norm`：层归一化；
- `depthwise_conv`：逐通道时间卷积；
- `pointwise_mlp`：逐时间点前馈网络；
- `gated_mlp`：门控前馈变换；
- `temporal_attention`：时间维自注意力；
- `spectral_gain`：FFT 低/高频增益调节；
- `trend_split`：趋势/残差分解；
- `channel_gate`：跨通道门控。

Agent 生成 1–6 个原语的有序图。每个原语通过残差更新组合，最后执行 LayerNorm。图中只允许上述注册原语，不接受 Python 表达式、导入语句、任意类名或外部代码。

### 2. 图结构进化

`propose_meta_operator_graph()` 根据误差描述提出初始计算图；`mutate_meta_operator_graph()` 支持插入、删除、替换和重排；`mutate_architecture()` 也能插入或变异元算子模块。元算子可作为 NAS 分支中的一个模块，与 LSTM、CNN、Transformer 等继续组合。

### 3. 误差驱动的研究假设

- 周期/相位误差：优先测试 `spectral_gain`、`temporal_attention`；
- 趋势漂移：优先测试 `trend_split`、`depthwise_conv`、`gated_mlp`；
- 跨变量相关性不足：优先测试 `channel_gate`、`gated_mlp`、`temporal_attention`；
- 弱点不明确：采用通用时域卷积、门控和注意力组合。

这些只是可证伪的候选假设，不是已证明的有效机制。

## 接口示例

```python
from atrea.nas import propose_meta_operator_graph, mutate_meta_operator_graph
from atrea.architecture import build_model

graph = propose_meta_operator_graph(weakness="periodic phase drift", seed=17)
spec = {
    "model_name": "NAS_Composed",
    "config": {
        "branches": [[graph, "lstm"]],
        "hidden": 32,
        "attention": "none",
        "attention_heads": 4,
        "fusion": "concat",
        "loss": "mae",
    },
}
model = build_model(spec, input_len=96, n_features=7, pred_len=24)
```

## 验证与边界

- 元算子图通过严格白名单验证后才允许实例化。
- PyTorch 前向、反向传播和非法原语拒绝测试已加入测试集。
- 完整项目测试：55 项通过。
- 本次尝试使用 ETTh1 进行元算子 CPU 训练时未在运行时限内完成，因此**不报告元算子 ETTh1 训练指标**。已有 V5 的 ETTh1 冒烟结果不能视为 V6 元算子的性能证据。
- 当前搜索仍是有限原语语法中的结构发现，不是任意数学函数空间的开放式发明。
- `promotion_allowed` 仍为 `False`。候选模型必须经过多随机种子、强基线、独立测试集、消融和发布门禁后才能申请晋级。

## 启动

```bash
python run_nas_etth1.py --population-size 8 --generations 2 --epochs 5 --batch-size 128 --seeds 11 22 33 --device auto --weakness "periodic phase drift"
```

建议先用较小种群和单随机种子检查流程，再逐步增加种群、训练轮数和随机种子。元算子可能增加计算成本，必须监控参数量、训练耗时和显存。
