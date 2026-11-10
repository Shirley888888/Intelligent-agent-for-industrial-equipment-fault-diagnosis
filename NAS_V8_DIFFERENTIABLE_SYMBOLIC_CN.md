# ATREA NAS V8：可微混合算子搜索与符号残差发现

## 1. V8 新增能力

### 1.1 可微混合算子搜索（DARTS-style）

新增 `darts_operator` 声明式算子节点。节点从白名单原语中选取 2–8 个候选操作（如深度卷积、逐点 MLP、频域增益、趋势分解、通道门控和时间注意力），并为候选操作建立可训练架构 logits：

\[
\alpha_i=\operatorname{softmax}(a)_i,\qquad
h'=\operatorname{LN}\left(h+\sum_i\alpha_i O_i(h)\right)
\]

架构权重与网络权重通过反向传播联合更新。`extract_darts_genotype(model, spec)` 将训练后的混合节点离散化为权重最大的原语，供后续独立训练、消融和复现实验使用。

**边界说明：** 当前版本实现的是 DARTS-style 可微混合操作与离散化，不是完整的双层（bilevel）DARTS：架构参数尚未采用独立验证集梯度更新，也没有对每个候选原语执行独立权重/架构优化器。因此不能将其描述为完整复现原始 DARTS 算法。

### 1.2 有界符号回归

新增 `atrea.symbolic_regression.search_symbolic_residual(inputs, residual)`，以及 `atrea.nas.search_symbolic_residual` 便捷入口。固定表达式语法包含常数、时间趋势、低阶时间多项式、正余弦项、输入变量、平方、绝对值、一阶差分和两两乘积。采用训练时间前缀上的贪心前向特征选择与岭回归拟合，再使用后续时间片段计算验证 MAE。

返回表达式、项系数、训练/验证样本数、验证误差及选择协议。整个过程不使用 `eval`、`exec` 或动态代码执行。该模块当前用于对预测残差进行可解释的候选修正关系发现；尚未自动写回预测模型或直接改变线上推理结果。

## 2. 搜索空间与安全约束

- DARTS 候选算子仅允许 `META_PRIMITIVES` 白名单。
- 候选数限制为 2–8，且不允许重复原语。
- 导出离散结构时只选择训练后 softmax 权重最大的原语。
- 符号回归只在固定数学语法内组合，不执行生成代码。
- 现有架构复杂度、分支数量、层数、损失函数和优化器白名单保持有效。

## 3. 建议实验协议

1. 固定数据版本、时间切分、归一化器和随机种子。
2. 将 V7 离散架构、V8 DARTS 混合架构及简单基线纳入同一验证协议。
3. 对 DARTS 节点记录训练前后架构 logits、softmax 权重和离散化结果。
4. 对符号回归记录训练前缀、验证后缀、表达式、系数和验证 MAE；不得在全量数据上拟合后再报告同一数据的误差。
5. 只有在多随机种子、重复训练和独立测试集上稳定优于基线，才允许晋级；单元测试通过或一次验证误差较低不代表性能提升。

## 4. API 示例

```python
from atrea.nas import (
    propose_darts_operator,
    extract_darts_genotype,
    search_symbolic_residual,
)

# DARTS mixed operation in a NAS_Composed branch
node = propose_darts_operator(seed=2028)

# After building and training the model, discretize the learned node
# discrete_spec = extract_darts_genotype(trained_model, mixed_spec)

# Discover an interpretable residual relation
# result = search_symbolic_residual(input_features, forecast_residual)
```
