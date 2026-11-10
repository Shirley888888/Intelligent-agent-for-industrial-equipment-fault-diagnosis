# ATREA NAS V9：双层可微架构搜索

## V9 目标

V8 的 DARTS 混合算子可以反向传播，但网络权重与架构 logits 仍在同一训练数据流中联合优化。V9 将二者拆分为两个优化步骤，并把架构更新与最终检查点选择使用的验证数据分开。

## 优化流程

1. **权重更新**：冻结 `architecture_logits`，仅使用训练批次更新候选算子和预测头等网络权重。
2. **架构更新**：冻结网络权重，仅使用架构验证窗口更新 `architecture_logits`。
3. **检查点选择**：使用另一段独立的验证窗口比较 epoch 检查点；该窗口不参与架构梯度更新。
4. **结构离散化**：训练结束后调用 `extract_darts_genotype`，将每个混合节点按最大 softmax 权重导出为注册算子图。

该实现属于**一阶交替双层优化（first-order alternating bilevel optimization）**，不是包含虚拟权重更新和二阶 Hessian-vector product 的完整二阶 DARTS。

## 数据隔离

- 原有 ETTh1 chronological 60/20/20 切分与仅训练前缀拟合的标准化流程保留。
- 原验证段再按时间顺序一分为二：前半段只用于架构 logits 更新，后半段只用于检查点选择。
- 最终测试尾段仍不参与 NAS 搜索、架构更新或检查点选择。

## 输出字段

训练记录增加 `bilevel_mode`、`bilevel_architecture_updates`、`architecture_validation_windows`、`checkpoint_selection_windows` 和 `discrete_architecture`。没有 DARTS 混合节点的普通候选仍按原训练流程工作，架构更新次数为 0。

## 局限与后续验证

一阶交替更新可验证架构参数确实由独立验证梯度驱动，但它不等价于完整二阶 DARTS。当前离散化采用最大 softmax 权重选择算子，仍需多随机种子、完整训练轮数、独立测试集以及与 V8/固定结构基线的配对比较，才能判断预测性能是否提升。不可用单次冒烟测试宣称模型更优。

## ETTh1 冒烟测试记录

CPU、seed=19、input_len=96、pred_len=24、batch_size=512、1 epoch 的 DARTS 训练成功完成：MAE=1.415672、RMSE=1.500390、参数量=2,075、耗时约 2.53 秒；架构更新 21 次。原验证段共 1,731 个窗口，其中 865 个用于架构更新，866 个用于检查点选择。最终导出离散结构为 `meta_operator( pointwise_mlp ) + linear`。

该实验仅证明训练管线、架构梯度更新、验证隔离和离散化导出均可运行。由于仅训练 1 epoch 且使用较小网络，不应将该 MAE 与 V8 其他配置直接比较，也不构成性能提升证据。
