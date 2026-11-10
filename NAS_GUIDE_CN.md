# ATREA 自主神经网络结构搜索（NAS）使用说明

## 本次升级

新增 `atrea/nas.py` 与 `NAS_Composed`，把搜索对象从固定模型名称扩展为可验证的网络拓扑：

- 并行分支及分支内模块顺序：CNN、LSTM、Transformer、Linear
- 每个分支 1–5 个模块，最多 3 个并行分支
- 隐藏宽度 16–256
- Attention：none / input / after_each / branch_end / fusion
- 特征融合：concat / mean / gated
- 损失函数：mae / huber / mae_rmse / horizon_weighted_mae
- Leave-one-component-out 消融候选

Agent 使用有限的声明式搜索空间，不执行 LLM 生成的 Python 代码。架构规格通过白名单、拓扑预算及实际参数量校验。

## 在宿主训练程序中调用

```python
from atrea.architecture import run_nas_search
from atrea.nas import build_loss
from atrea.architecture import build_model

def train_evaluate(spec, seed, loss_fn):
    # 这里接入项目真实 ETTh1/工业油温数据加载、训练和验证逻辑。
    # 必须保证所有候选使用相同 chronological split 与预处理。
    model = build_model(spec, input_len=96, n_features=7, pred_len=24)
    from atrea.nas import build_optimizer, build_scheduler
    optimizer = build_optimizer(model, spec["config"])
    scheduler = build_scheduler(optimizer, spec["config"])
    # for epoch in ...:
    #     pred = model(x_batch)
    #     loss = loss_fn(pred, y_batch)
    #     loss.backward(); optimizer.step()
    #     if scheduler is not None:
    #         if spec["config"].get("scheduler") == "plateau":
    #             scheduler.step(validation_loss)
    #         else:
    #             scheduler.step()
    # metrics 必须由冻结验证集上的真实预测计算，不能返回模拟值。
    return {"metrics": {"mae": real_validation_mae,
                        "rmse": real_validation_rmse}}

def ablation_evaluate(spec, seed, loss_fn):
    # 与主候选完全相同的数据切分、训练预算和评估指标
    return train_evaluate(spec, seed, loss_fn)

result = run_nas_search(
    train_evaluate=train_evaluate,
    ablation_evaluate=ablation_evaluate,
    split_hash="sha256-of-frozen-train-validation-split",
    dataset="ETTh1",
    population_size=12,
    generations=2,
    seeds=[11, 22, 33],
    metric="mae",
)
```

`real_validation_mae` / `real_validation_rmse` 是宿主训练代码计算的值，以上仅展示回调接口，不是可直接运行的训练脚本。不要将占位变量或合成指标当作真实实验结果。

## 输出与选择逻辑

`run_nas_search` 返回：

- `experiments`：逐架构、逐随机种子的不可混淆实验记录
- `generations`：每代候选数、成功次数、最优验证指标及幸存者数
- `ranked_candidates`：跨随机种子聚合均值后的架构排序
- `best_candidate`：当前验证集排名第一的候选
- `ablation_experiments`：结构消融结果
- `component_effects`：消融相对完整结构的指标变化及解释
- `promotion_allowed=False`：NAS 研究结果不会直接部署

## 必须遵守的实验边界

1. 只在训练集拟合参数；超参数和架构选择只使用验证集。
2. 同一轮候选必须使用相同数据切分、预处理、训练预算和随机种子集合。
3. 不得用最终测试集挑选架构；最终测试集仅在搜索方案冻结后评估。
4. 以多个随机种子的均值排名，并同时报告标准差；单次最好成绩不能证明结构有效。
5. 消融候选与父模型进行配对比较。删除某组件后验证指标不恶化，才将其列为可移除候选。
6. 最终上线仍需通过独立冻结测试、数据泄漏检查、资源预算、影子运行、人工审批及回滚流程。
7. 当前 NAS 搜索空间是受控的可组合空间，不是无限制的任意代码生成器；新的算子类型必须先实现、测试并登记。

## 测试

```bash
python -m unittest tests_agent.test_architecture_evolution tests_agent.test_autonomous_research_engine tests_agent.test_nas -v
```

## NAS V4：结构创新与实验驱动搜索

在 V3 组合拓扑基础上，V4 新增以下已登记、可执行算子：

- `trend_residual`：滑动趋势与短期残差的可学习重组；
- `frequency_mixer`：沿时间轴进行 rFFT 低/高频分解及可学习频段重加权；
- `cross_var`：每个时间步将输入通道作为变量 token，执行跨变量注意力交互后聚合。

搜索同时覆盖训练配方：`adamw/adam/rmsprop`、学习率、weight decay、`none/cosine/plateau` 调度器，以及 `mae_slope`、可调权重的 `mae_rmse` 等损失。`build_optimizer(model, spec['config'])` 与 `build_scheduler(optimizer, spec['config'])` 可直接构建对应训练组件；宿主回调仍须负责按 epoch 正确 step scheduler。

新增遗传交叉：从两个已验证父代重组分支和训练配方。消融记录会保留父代架构指纹及同随机种子的父代验证指标，`infer_structure_hypotheses` 将消融差异转成下一轮可检验的结构假设。`run_nas_search` 输出 `next_research_hypotheses` 和 `next_generation_seed_population`。研究循环会从历史实验记忆中取同数据集的 NAS 拓扑作为未排序初始素材；这些旧拓扑仍需在本轮冻结切分上重新训练，不直接继承旧成绩。

### 解释边界

这些是受控算子空间内的自动结构发现，不是自动编写任意 PyTorch 代码。`cross_var` 在首层直接对应观测变量；若放在后续层，则交互对象是潜在通道。消融只产生候选假设，单次消融不能证明因果关系；需要跨随机种子重复、配对比较及独立测试集复核。优化器和调度器搜索只有在宿主 `train_evaluate` 回调实际调用上述构建器时才生效。


## 使用项目自带 ETTh1 真实数据直接运行 NAS

工程新增 `atrea/nas_training.py`，复用项目的 ETTh1 数据校验、按时间顺序 60/20/20 切分及仅用训练前缀拟合的标准化流程。所有候选共享同一训练/验证窗口；NAS 只读取验证集指标，最终测试尾段不会参与搜索。

```bash
python run_nas_etth1.py --population-size 8 --generations 2 --epochs 5 --batch-size 128 --seeds 11 22 33 --device auto
```

结果默认保存到 `output/nas/etth1_nas_results.json`。可先用 `--population-size 2 --generations 1 --epochs 1 --seeds 11 --device cpu` 做真实数据流程冒烟验证。正式研究应增加种子、训练预算与搜索代数，并在结构冻结后另行执行最终测试。

## V7：数学结构发现

V7 在 V6 有序元算子图上增加受控数学方程搜索，支持 `residual_add`、`gated_residual`、`multiplicative_modulation`、`parallel_residual` 和 `trend_spectral_mix`。每个数学算子有独立主路径和可选门控路径，可被 Agent 搜索、变异并进行路径级消融。详细方程、使用方式与验证边界见 `NAS_V7_MATHEMATICAL_DISCOVERY_CN.md`。

## V8：可微混合算子与符号残差搜索

V8 新增 `darts_operator`：在白名单原语间使用 softmax 架构权重形成可微混合算子，训练后可通过 `extract_darts_genotype` 导出离散结构。同时新增安全的符号残差回归，用固定数学语法发现趋势、周期及变量交互项，并以时间顺序留出段评估。详细使用方式和限制见 `NAS_V8_DIFFERENTIABLE_SYMBOLIC_CN.md`。当前实现为 DARTS-style 联合梯度搜索，尚非完整双层 DARTS；符号回归结果也不会自动部署到预测模型。


## V9：一阶交替双层 DARTS

V9 将 DARTS 架构 logits 与网络权重分开优化：训练批次更新网络权重，独立架构验证窗口更新架构 logits，另一个验证窗口负责检查点选择。训练结束会导出离散化结构。该实现是一阶交替双层优化，并非完整二阶 DARTS；详见 `NAS_V9_BILEVEL_DARTS_CN.md`。

## 失败驱动的候选模型进化（新增）

失败驱动实验协调器位于 `dual_evolution/failure_evolution.py`。它可接收 NAS/训练器产生的候选结构，执行“训练—门禁—失败诊断—下一候选提案”的有界闭环。训练器和门禁由宿主通过回调注入，确保候选模型与生产基线使用相同、冻结且无泄漏的评估样本。

注意：该模块只负责实验编排和经验积累，不会自行猜测门禁阈值，不会调用生产发布接口。门禁 `PASS` 后仅进入独立影子验证，不代表已上线。具体接口和示例见 `FAILURE_DRIVEN_EVOLUTION_CN.md`。
