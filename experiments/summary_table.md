# Four-arm online learning experiment summary

Controlled concept drift is injected only into the loaded ndarray after the configured cutoff; the source CSV is unchanged.

| Arm | Overall MAE | Overall MSE | Worst 10-episode MAE | Retrain triggers | Activations | Rollbacks | Recovery (episodes) | Mean decision steps / episode | Complete episodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Arm 1 - Frozen Model | 1.670403 | 5.366424 | 2.782199 | 0 | 0 | 0 | 15 | 1.00 | 108 |
| Arm 2 - Periodic Retrain | 1.638660 | 5.222721 | 2.731238 | 3 | 3 | 0 | 11 | 1.00 | 108 |
| Arm 3 - Feedback-Triggered ReAct | 1.650223 | 5.278552 | 2.741295 | 8 | 1 | 0 | 12 | 26.18 | 108 |
| Arm 4 - Feedback-Triggered No-Gate | 1.640981 | 5.206774 | 2.741295 | 3 | 3 | 0 | 11 | 26.05 | 108 |

Recovery is measured from the first 10-episode rolling MAE above baseline×1.2 to the first later value at or below baseline×1.1. N/A means no qualifying recovery was observed before replay ended.
