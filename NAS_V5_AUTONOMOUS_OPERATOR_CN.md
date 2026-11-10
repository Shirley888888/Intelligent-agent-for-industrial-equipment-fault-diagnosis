# ATREA NAS V5：自主算子创新与多目标结构发现

## 本版新增

1. **AmplitudePhaseBlock**：新增可学习频谱幅值—相位算子。对输入序列做 rFFT，分别学习每个潜在通道的频率幅值增益和相位旋转，再通过 irFFT 回到时域并以残差方式融合。
2. **误差诊断驱动提案**：`propose_operator_innovations()` 根据周期/相位、趋势漂移、跨变量相关等弱点提出可证伪的结构假设，并生成受白名单约束的声明式架构配置。
3. **多目标 Pareto 搜索**：新增 `pareto_front()`，同时考虑 MAE、RMSE、参数量和训练耗时（后两项由训练回调提供时纳入）；进化保留优先覆盖 Pareto 前沿，再按主指标补足种群。
4. **训练耗时记录**：ETTh1 runner 记录参数量、训练时间、优化器和调度器等信息。
5. **研究闭环接入**：研究周期可将诊断弱点传入 NAS，并将 Pareto 候选、消融结果和后续假设保存到研究计划。

## 启动示例

```bash
python run_nas_etth1.py --population-size 8 --generations 2 --epochs 5 --batch-size 128 --seeds 11 22 33 --device auto --weakness "periodic phase drift and cross-variable interaction"
```

## 研究依据与边界

- Cao et al., *Automatic selection of the best neural architecture for time series forecasting*, Nature Communications, 2026. DOI/页面：https://www.nature.com/articles/s41467-026-73687-9 。启发：复合模块顺序搜索、多目标 Pareto 前沿和应用偏好选择。
- Xu et al., *FreAPNet: Learnable Amplitude–Phase Parameterization and Dual-Path Frequency Learning for Multivariate Time Series Forecasting*, 2026. DOI：https://doi.org/10.1002/cpe.70675 。启发：可学习频谱幅值和相位参数化。

以上论文提供设计启发，不代表 ATREA 已复现论文结果。AmplitudePhaseBlock 是本项目新增的研究候选算子，必须通过统一数据切分、多随机种子、基线对比、独立测试和消融实验才能判断是否有效。

## 安全和晋级规则

- Agent 只能提出声明式配置，不能生成并执行任意 Python 源码。
- 搜索排名只使用训练/验证流程；最终测试集不能参与搜索、调参或 Pareto 选择。
- Pareto 前沿是多目标非支配集合，不等于每个候选都比基线准确。
- `promotion_allowed` 仍为 `False`；未经独立测试和发布门禁不得自动替换线上模型。
