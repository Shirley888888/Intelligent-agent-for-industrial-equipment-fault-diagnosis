# ETT-small 数据来源、署名与许可说明

## 官方数据源

- GitHub 仓库：https://github.com/zhouhaoyi/ETDataset
- 原始 CSV 目录：https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small
- 固定源提交：`1d16c8f4f943005d613b5bc962e9eeb06058cf07`（2021-06-04）。
- 原始文件下载地址模板：`https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/{ETTh1,ETTh2,ETTm1,ETTm2}.csv`。
- `download_ett.py` 对四个原始 CSV 固定 Git blob SHA-1；下载完成还会计算 SHA-256，并检查列名、数值有限性、采样间隔和最少行数。
- 数据任务：电力变压器油温（`OT`）多变量时间序列预测。
- 数据介绍：研究团队说明数据来自与北京国网富达科技发展公司合作建立的真实运行平台，覆盖两个地区、约两年，包含油温及六种负载特征。
- 建议引用：Zhou, H. et al. *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*. AAAI 2021.

## 许可

上游仓库 LICENSE 文件标注 **Creative Commons Attribution-NoDerivatives 4.0 International (CC BY-ND 4.0)**。

本项目的 `download_ett.py` 只下载未经改写的原始 CSV，并在 `data/ett/source_manifest.json` 记录来源 URL 与 SHA-256。不要删除原始数据署名或许可信息。

CC BY-ND 对分享改编材料有限制。本仓库因此不把在线下载到的 ETTm1/ETTm2 等重新处理 CSV 作为本项目生成物分发。对训练后模型权重是否构成可分享的改编材料，不在本项目中作法律结论；公开发布训练权重、派生数据或商业使用前，请核对许可条款并在必要时取得权利方许可。

## 公开范围边界

上游 README 当前将 ETT-small 标记为公开；ETT-large（39 台/39 站）和 ETT-full（69 台/39 站）仍标记为未公开。不要将后两者写成已取得数据。

## 数据组

| 文件 | 采样间隔 | 语义预测跨度（96 输入 / 24 输出） |
|---|---:|---:|
| ETTh1 / ETTh2 | 60 分钟 | 输入 96 小时，预测 24 小时 |
| ETTm1 / ETTm2 | 15 分钟 | 输入 24 小时，预测 6 小时 |

小时组与 15 分钟组必须分别训练，不能把不同采样间隔的行直接拼接。
