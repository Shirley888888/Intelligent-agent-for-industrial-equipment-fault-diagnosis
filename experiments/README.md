# ReAct 在线学习四臂实验

## 运行
在项目根目录执行 `python run_experiments.py`，或执行 `python experiments/run_experiments.py`。默认使用 NumPy Linear 后端，不需要 GPU。

## 唯一配置入口
请直接编辑本目录的 `config.yaml`。它是 JSON-compatible YAML，避免额外依赖；包含固定时间切分、seed=42、监控阈值、Gate 的 5%/2% 规则、Periodic 的 K 和漂移参数。驱动运行时保留此文件，清理并重新生成实验结果。

## 漂移协议
`ETTh1.csv` 只读。驱动在加载后的 ndarray 副本中，从 `online_start + 60*24` 起仅对 OT 通道施加 +2℃ 均值偏移及 1.5 倍偏差幅度扩张，四臂共用同一副本。漂移变换及参数写入 `drift_manifest.json`。

## 实验臂
- `frozen/`：M0 冻结。
- `periodic/`：每 30 个完整反馈 Episode 训练并直接发布候选。
- `agent/`：反馈触发 ReAct，启用双条件 Gate。
- `no_gate_ablation/`：反馈触发训练，明确跳过 Gate。

各臂保存 predictions、feedback、monitor、retrain events、version events、current pointer、summary、候选/版本产物及 react trace。汇总表、PNG、模型更新说明、trace 示例和结论位于本目录根部。
