# ATREA 自主研究与模型进化升级（V2）

## 本轮真实代码改动

1. **ReAct 可审计轨迹**：`atrea/react.py` 记录 observe / think / act / reflect 阶段的动作摘要、观测结果、产物 ID 和事件 SHA-256；不保存私有思维链。
2. **多来源论文发现**：`atrea/papers.py` 同时查询 arXiv 与 Semantic Scholar，按关键词匹配、时效和可识别方法线索形成候选排序；可通过 `SEMANTIC_SCHOLAR_API_KEY` 提高接口稳定性。排序分数只用于检索，不代表论文质量或模型全网最优。
3. **实验记忆隔离**：SQLite 增加 `memory_scope` 兼容迁移。A1/A2 结果只写入各自消融 JSON 文件；A2 不写入可复用实验/论文记忆；A3 只检索和积累 A3 自己的实验与经验模式。
4. **经验归纳更严格**：仅在同一数据集、同一时间切分、存在相同随机种子的配对实验中归纳模型比较模式，避免把不配对的结果误写成经验规律。
5. **残差分析**：训练运行器提供对齐的 `y_true` / `y_pred` 时，自动计算 MAE、RMSE、偏差、残差标准差和最大绝对误差。
6. **发布安全门禁**：候选实验、门禁结论、实验数据库和论文报告先持久化，再执行激活回调；缺少冻结测试、基线比较、数据泄漏检查、回滚准备等检查时阻止激活。
7. **调度器可靠性**：每日北京时间 08:00；失败会记录错误并按退避间隔重试，成功后当天不重复运行；状态文件原子替换。
8. **真实 Chronos-2 训练接入器**：`atrea/chronos2_runner.py` 调用仓库现有 `train_chronos2_ett.py`，使用其真实数据校验、Chronos-2 LoRA/full fine-tuning、冻结时间切分和评估 manifest，返回零样本基线与微调候选的结构化实验记录。
9. **每日任务模板**：`atrea/daily_job_template.py` 将调度器连接到可注入的实验运行器和论文/经验联合规划器。

## 每日运行

1. 复制 `configs/chronos2_daily.example.json` 并按数据、显存和预测目标修改配置。
2. 安装项目基础依赖；Chronos-2 环境另按 `requirements_chronos.txt` 安装兼容版本，并确认 ETTh1/ETTh2 数据已下载。
3. 提供一个 Python 模块工厂：
   - `build_runner()` 返回 `Chronos2ExperimentRunner` 或其他真实训练运行器；本项目内置 `atrea.chronos2_runner:build_runner_from_env`。
   - `build_hypothesis_builder()` 返回函数 `hypothesis_builder(papers, memory)`，负责把论文和历史经验转换成可执行的 `plan`，Chronos-2 参数候选放在 `plan.chronos2_config`。
4. 设置环境变量 `ATREA_HYPOTHESIS_BUILDER=你的模块:build_hypothesis_builder`，然后运行 `run_atrea_daily.example.bat`。也可在 Windows 任务计划程序中每天 08:00 启动该脚本。

示例规划结果结构：

```python
{
  "statement": "在相同时间切分下，较长上下文是否降低跨站点油温预测误差？",
  "rationale": "论文方法线索与历史实验经验共同支持该可证伪假设。",
  "plan": {
    "models": ["Chronos-2-zero-shot", "Chronos-2-finetuned"],
    "same_split": True,
    "same_preprocessing": True,
    "chronos2_config": {
      "context_length": 768,
      "num_steps": 1200,
      "learning_rate": 0.00001,
      "finetune_mode": "lora"
    }
  }
}
```

## 三臂消融实验

调用 `run_three_arm_ablation` 时，真实运行器必须遵守各臂约束：

- A1：固定架构，不联网检索，不读取经验。
- A2：联网检索，必须真实改变架构，不读取、不写入可复用实验记忆。
- A3：联网检索，必须真实改变架构，读取并积累 A3 自己的实验经验。

`Chronos2ExperimentRunner` 当前负责 Chronos-2 同架构微调，不会谎称修改了 Chronos-2 网络结构，因此它**不能单独充当 A2/A3 的架构搜索运行器**。A2/A3 还需接入能够实例化不同架构（例如已实现且可训练的 PatchTST、iTransformer、TimesNet 或项目自定义模型）的训练适配器。没有真实架构差异时，消融函数会拒绝把该轮记为架构修改实验。

## 仍需宿主侧提供 / 尚未声称完成

- LLM 规划器工厂：将论文摘要与 A3 经验转为可执行模型/参数计划。当前不绑定任何未配置的 API 密钥或模型服务。
- A2/A3 多架构训练适配器：Chronos-2 本身的训练入口是固定模型家族的微调，不支持任意论文架构自动改写。
- 正式生产发布回调：将候选 checkpoint 接入已有模型版本注册、灰度、回滚系统。默认 `allow_activation=False`，不会因论文分数自动替换线上模型。
- 真实运行结果：测试通过只说明接口和逻辑通过，不代表已在用户 GPU 上完成新的 Chronos-2 训练。

## 验证

本次新增测试覆盖：记忆隔离、A3 跨轮记忆读取、调度失败重试与当日幂等、残差诊断、ReAct 轨迹和报告持久化。完整测试结果以随包 `TEST_RESULTS_V2.txt` 为准。
