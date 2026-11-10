# Changelog — ATREA NAS V7

- 新增五类受控数学方程模板：残差加法、门控残差、乘性调制、双路径并行残差、趋势—频谱混合。
- 新增独立主路径/门控路径的原语搜索与变异。
- 新增错误模式驱动的数学算子候选生成，并接入 NAS 第一代候选注入。
- 新增数学方程、主路径和门控路径的消融实验生成。
- `NASComposedModel` 新增 `MathematicalOperatorBlock` 安全编译器；不执行任意生成代码。
- 搜索模式标识更新为 `hypothesis_driven_mathematical_operator_nas_v7`。
- 修复 NAS 多分支 forward 中重复执行分支编码的问题。
- 保留模型晋级门禁关闭状态；没有在未完成独立评估时自动替换 incumbent。
