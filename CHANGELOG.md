# 版本切换一致性修复

修改基线：最新上传的 `online_loop_minimal.zip`。保留类名、函数签名、数据结构、模型后端和原实验记录。

| 位置 | 修改 |
|---|---|
| `VersionManager.activate` | 日志先写、指针最后提交；异常时截断部分日志或删除首次新建日志，清理指针临时文件；成功提交后不执行额外 I/O |
| `VersionManager.publish` | 激活失败删除本次新建的未激活版本目录，候选与历史版本保留，可重新发布 |
| `simulate` | 门禁通过后在提交前加载并试推候选，提交后直接使用准备好的模型句柄，避免加载失败导致版本与权重不符 |
| `test_loop.py` | 新增日志失败、回滚失败、日志部分写入、指针替换失败、初始化失败及重试、模拟器失败保留与正常升级路径测试 |
| Tool/设计说明 | 同步提交顺序、副作用及单写者异常恢复范围 |

本次故障测试的门禁分数是受控测试输入，不是模型精度收益。普通 I/O 异常恢复不等于断电/进程终止时的跨文件原子性；清理操作自身失败时会显式报错。

# v2 针对性修改清单（保留原记录）

修改基线：本次上传的 ETTh1_Online_Learning_Minimal_Loop(1).zip。
保留原函数名、类名、数据字段和 NumPy / LSTM 两条后端；未重写 models.py 或修改 ETTh1.csv。

| 前面要求 | 本次落地位置 | 修改与验证 |
|---|---|---|
| 1. Tool 清单与 I/O | tool_catalog.py、tool_catalog.json、minimal_loop.py 的 TypedDict | 从实际签名导出类型、默认值、输出结构和约束；公开 tools 命令 |
| 2. Feedback Episode | FeedbackEpisode、predict_tool、complete_feedback、FeedbackEpisode_example.json | 保留全部字段；检查历史未篡改、预测形状、真值时间逐点对齐及成熟条件 |
| 3. Feedback Monitor | FeedbackMonitor.run / reset | 每次返回完整固定结构，最近 N 次滚动 MAE；非重叠块作触发依据；批量反馈和激活切换自动重置 |
| 4. Train Candidate | TrainConfig.validate、train_candidate_tool、configs/ | 配置 JSON 与完整 CLI；拟合与独立验证分离；显式有限值和时间边界检查；新增 GPU 执行检查入口 |
| 5. Model Gate | GateConfig、model_gate_tool、validate_candidate | 近期改善和历史容差均量化；绑定候选 hash、数据快照、现役 hash 和 activation_id；非法指标拒绝且返回合法 JSON |
| 6. 版本与回滚 | VersionManager.publish / activate / list_versions | 完整产物暂存后获取 M 编号；指针原子切换；旧 gate 不可在回滚后复用；恢复前检查权重/配置/基线/时钟 |
| 7. ETTh1 模拟 | simulate、run_config.json、results_*_v2 | 严格时间前缀、延迟 truth、训练/验证/两门禁隔离；训练失败继续使用 M0；v1 原记录保留 |
| 无未来泄露验证 | check_observed / check_manifest、test_loop.py | 将可被优化模式移除的 assert 改为显式 ValueError；测试未来扰动前缀不变及 python -O 校验 |

## 改动注释

代码使用 `modify:` 标记，例如 explicit-validation、prefix-contract、rolling-monitor、publication-binding、atomic-version、full-cli。
所有旧断言边界改用统一 require()，即使 Python 以 -O 运行也不绕过校验。对于非法外部输入，异常类型从 AssertionError 统一改为 ValueError。

## 兼容性与结果边界

- 原 CLI simulate / rollback 仍可使用；新增配置参数、tools / versions 子命令。
- 原生成的保存版本可通过相对目录加载；新发布要求使用新的门禁绑定字段。
- Monitor 等待下一检查块时，也返回已有最近 N 样本 MAE；evaluated_blocks 表示本次是否进行了新的触发判断。
- 按默认配置重跑的参考 MAE 与 v1 在数值容差内一致。新增验证不能表述为 LSTM 在线精度提升。
- 本次未执行 GPU 分支；smoke_torch.py 在用户已有 PyTorch 环境中进行实际执行检查。
