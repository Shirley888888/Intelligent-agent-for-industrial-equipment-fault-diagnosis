# ReAct 在线学习四臂实验（GPU / LSTM 变体）

与 `experiments/` 的默认记录逐项同参数，唯一差异是后端与设备：`backend=torch`、`model_name=LSTM`、`device=cuda:0`。
切分、seed、监控阈值、Gate 5%/2%、Periodic K 与漂移参数全部相同，因此两目录可直接对照后端差异。

## 运行

```powershell
conda activate etth1_gpu
$env:EXPERIMENTS_CONFIG = "experiments_gpu/config.yaml"
$env:EXPERIMENTS_ROOT   = "experiments_gpu"
python run_experiments.py
```

`EXPERIMENTS_CONFIG` / `EXPERIMENTS_ROOT` 是驱动的两个环境变量开关；不设置时行为与原先完全一致（读 `experiments/config.yaml`、写 `experiments/`）。
未使用非默认配置的运行禁止写入默认 `experiments/` 目录，避免覆盖 CPU 记录。

## 执行环境

本机 `etth1_gpu`：Python 3.11.16、torch 2.7.1+cu128、RTX 5060、matplotlib 3.11.1。每次运行时长为四臂共 4 次引导训练与 11 次候选训练。

## 结果边界

- 结果是同一受控漂移协议下的 **LSTM 后端**回放，不是 NumPy Linear 参考后端的重复，两者不可混用或相互替换。
- Arm 3 / Arm 4 各出现 1 次自动回滚，来自 `simulate` 内父版本反事实比较策略：当前版本在成熟窗口上持续劣化且父版本明显更优时安全回退。
- Gate 收益需按数值判断，不得依据机制预设；结论原文见 `conclusion.md`。
