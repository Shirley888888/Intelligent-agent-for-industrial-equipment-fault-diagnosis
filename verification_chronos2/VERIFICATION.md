# Chronos-2 油温预测大模型接入验收记录

## 测试结果

本次针对数据下载/校验、Chronos-2 训练编排与 Agent 接口，以及项目既有 Agent 和双自进化模块，执行了 6 组测试：

| 测试套件 | 用例数 | 通过 | 失败 |
|---|---:|---:|---:|
| tests_chronos | 11 | 11 | 0 |
| tests_ett | 5 | 5 | 0 |
| tests_agent | 29 | 29 | 0 |
| tests_integration | 8 | 8 | 0 |
| tests_dual | 12 | 12 | 0 |
| tests_dual_task | 9 | 9 | 0 |
| **合计** | **74** | **74** | **0** |

此外，ETTh1 单文件 preflight 通过：字段、源文件哈希、采样间隔、96 点历史上下文、24 点（小时级 24 小时）预测和 60/10/10/20 时间切分均通过。

## 真实数据来源核验

- 官方仓库：`https://github.com/zhouhaoyi/ETDataset`
- 固定提交：`1d16c8f4f943005d613b5bc962e9eeb06058cf07`
- 本包 ETTh1：17420 行，1 小时频率；SHA-256 和 Git blob SHA-1 与本地 manifest 一致。
- ETTh2 / ETTm1 / ETTm2 的 Git blob SHA-1 已从官方 GitHub Contents API 固定到下载器和 `data/ett/source_manifest.json`；它们需要在可访问 GitHub 的训练机执行下载。

## 不可声称已完成的部分

当前运行容器为 Python 3.13.5、PyTorch 2.10.0+cpu，CUDA 不可用，且无出站 DNS/网络；`chronos`、`transformers`、`peft`、`accelerate` 不存在。因此未下载 `amazon/chronos-2` 权重，未运行真实 LoRA 微调，也未生成 Chronos-2 微调 Checkpoint 和模型性能对比。

Chronos 测试使用 fake/mock pipeline，只验证训练脚本的数据构造、split、候选门禁、quantile 指标和 Agent Episode 接口。此类 mock 的 MAE 数值不是性能结果。

## 用户 GPU 机器执行顺序

1. 在已有 CUDA PyTorch 环境中安装 `requirements_chronos.txt`。
2. 执行 `download_ett.py` 下载并逐文件校验四个官方 CSV。
3. 运行两个频率的 `--preflight-only`。
4. 先运行 `run_chronos2_oil_smoke_windows.bat` 做 20-step smoke。
5. 观察显存、loss、candidate gate 后再运行 `run_chronos2_oil_foundation_training_windows.bat` 进行正式 1000-step LoRA。
6. 审核测试集、候选门禁和模型 manifest；通过后还需影子验证与人工批准，不能自动激活。
