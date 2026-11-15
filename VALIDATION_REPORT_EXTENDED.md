# ATREA 扩展验证报告（云端 RTX 4080 SUPER）

## LSTM 多 seed（默认门控，8 个 seed）
- seed 0: baseline 1.8170, prequential 1.2707, 改进 +30.1%, upgrades=0
- seed 1: baseline 1.5091, prequential 1.3465, 改进 +10.8%, upgrades=0
- seed 2: baseline 1.6227, prequential 1.3740, 改进 +15.3%, upgrades=1
- seed 3: baseline 1.5831, prequential 1.3469, 改进 +14.9%, upgrades=1
- seed 4: baseline 1.5726, prequential 1.3606, 改进 +13.5%, upgrades=1
- seed 5: baseline 1.6796, prequential 1.3769, 改进 +18.0%, upgrades=1
- seed 6: baseline 1.7069, prequential 1.4655, 改进 +14.1%, upgrades=1
- seed 7: baseline 1.5237, prequential 1.2674, 改进 +16.8%, upgrades=0
- **汇总**: n=8  baseline 1.6268±0.1033  prequential 1.3511±0.0631  improvement +16.7%±5.8%  upgrades=5

## TCN 多 seed（默认门控，3 个 seed）
- seed 0: baseline 1.8047, prequential 1.3623, 改进 +24.5%, upgrades=1
- seed 1: baseline 1.9912, prequential 1.2870, 改进 +35.4%, upgrades=0
- seed 2: baseline 1.7580, prequential 1.3409, 改进 +23.7%, upgrades=1
- **汇总**: n=3  baseline 1.8513±0.1234  prequential 1.3301±0.0388  improvement +27.9%±6.5%  upgrades=2

## LSTM 放宽门控升级生命周期（3 个 seed）
- seed 0: baseline 1.8170, prequential 1.2707, 改进 +30.1%, upgrades=0
- seed 1: baseline 1.5091, prequential 1.2560, 改进 +16.8%, upgrades=1
- seed 2: baseline 1.6227, prequential 1.3740, 改进 +15.3%, upgrades=1
- **汇总**: n=3  baseline 1.6496±0.1557  prequential 1.3003±0.0643  improvement +20.7%±8.1%  upgrades=2

## 升级后 M1 基线
- seed 0: 版本 ['M0'], 当前 M0 (activation_id=0, action=bootstrap)
- seed 1: 版本 ['M0', 'M1'], 当前 M1 (activation_id=1, action=upgrade)
- seed 2: 版本 ['M0', 'M1'], 当前 M1 (activation_id=1, action=upgrade)