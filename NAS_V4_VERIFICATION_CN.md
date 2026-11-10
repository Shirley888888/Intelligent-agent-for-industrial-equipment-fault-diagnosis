# ATREA NAS V4 实现与验证记录

## 实现范围

1. 新增 `trend_residual`、`frequency_mixer`、`cross_var` 三种可训练结构算子。
2. 搜索空间扩展至分支组合、增删层、模块替换、分支交叉、Attention 位置、融合方式、损失函数、优化器、学习率、weight decay、scheduler、dropout 与损失权重。
3. 新增可调用的 optimizer/scheduler builder；NAS 真实 ETTh1 runner 已使用候选配置创建优化器和调度器。
4. 消融覆盖结构模块、Attention、融合方式及损失/优化器配方对照。
5. 消融记录带有父代架构指纹及同随机种子父代指标，转化为明确标注为 preliminary 的结构假设。
6. 研究循环将假设和下一代候选种群保存为 candidate patterns；下一轮重新训练历史候选后才允许参与排名。
7. 新增真实 ETTh1 数据训练适配器及命令行入口 `run_nas_etth1.py`。统一 chronological 60/20/20 切分，标准化参数仅由训练前缀拟合；NAS 仅使用验证段，最终测试尾段不进入搜索。

## 验证结果

- `tests_agent`: 50 passed
- `tests_integration`: 8 passed
- `tests_dual_task`: 9 passed
- `tests_ett`: 5 passed
- `tests_dual`: 12 passed
- `tests_chronos`: 11 passed
- 根目录测试：29 passed，1 skipped（环境没有 CUDA，跨设备测试跳过）
- 合计：124 passed，1 skipped。

另完成真实 ETTh1 数据单候选训练冒烟测试：CPU、1 epoch、batch size 512、1731 个验证窗口；候选结构为单层 Linear NAS 分支，验证 MAE=1.2557、RMSE=1.4171（训练集拟合标准化后的目标尺度）。这仅证明真实数据加载、切分、训练、验证与指标回传链路可运行，不是 NAS 搜索结果，也不能据此声称模型优于基线。

## 运行正式 NAS

```bash
python run_nas_etth1.py --population-size 8 --generations 2 --epochs 5 --batch-size 128 --seeds 11 22 33 --device auto
```

正式实验需要更大的搜索预算、多个随机种子、稳定的训练预算，并在搜索结束、结构冻结后独立运行最终测试。默认 `promotion_allowed=False`，不会自动部署候选模型。
