# Model update log

Times are ETTh1 dataset-local timestamps. Gate scores are the recent/historical incumbent and candidate MAEs when a gate was evaluated.

- **Arm 2 - Periodic Retrain** — 2018-04-08T23:00:00: M0 → M1; action=upgrade; gate=not run (ablation policy); baseline MAE=1.072218.
- **Arm 2 - Periodic Retrain** — 2018-05-08T23:00:00: M1 → M2; action=upgrade; gate=not run (ablation policy); baseline MAE=0.985555.
- **Arm 2 - Periodic Retrain** — 2018-06-07T23:00:00: M2 → M3; action=upgrade; gate=not run (ablation policy); baseline MAE=1.308495.
- **Arm 3 - Feedback-Triggered ReAct** — 2018-05-11T23:00:00: M0 → M1; action=upgrade; gate=recent 2.261480234217762 → 1.9654820436873346; historical 2.2883929022772493 → 1.5607358717375133; baseline MAE=1.965482.
- **Arm 3 - Feedback-Triggered ReAct** — 2018-06-22T23:00:00: M1 → M0; action=rollback; gate=not run (ablation policy); baseline MAE=1.517047.
- **Arm 3 - Feedback-Triggered ReAct** — 2018-04-06T23:00:00: candidate C1 rejected/not activated; recent_ok=False, historical_ok=True.
- **Arm 3 - Feedback-Triggered ReAct** — 2018-05-18T23:00:00: candidate C3 rejected/not activated; recent_ok=False, historical_ok=True.
- **Arm 4 - Feedback-Triggered No-Gate** — 2018-04-06T23:00:00: M0 → M1; action=upgrade; gate=not run (ablation policy); baseline MAE=1.035575.
- **Arm 4 - Feedback-Triggered No-Gate** — 2018-04-13T23:00:00: M1 → M2; action=upgrade; gate=not run (ablation policy); baseline MAE=1.919881.
- **Arm 4 - Feedback-Triggered No-Gate** — 2018-05-18T23:00:00: M2 → M3; action=upgrade; gate=not run (ablation policy); baseline MAE=1.929371.
- **Arm 4 - Feedback-Triggered No-Gate** — 2018-06-22T23:00:00: M3 → M2; action=rollback; gate=not run (ablation policy); baseline MAE=1.919881.
