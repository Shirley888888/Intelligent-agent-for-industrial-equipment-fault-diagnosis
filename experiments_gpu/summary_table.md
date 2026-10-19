# Four-arm online learning experiment summary

Controlled concept drift is injected only into the loaded ndarray after the configured cutoff; the source CSV is unchanged.

| Arm | Overall MAE | Overall MSE | Worst 10-episode MAE | Retrain triggers | Activations | Rollbacks | Recovery (episodes) | Mean decision steps / episode | Complete episodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Arm 1 - Frozen Model | 1.582204 | 4.864989 | 2.621977 | 0 | 0 | 0 | 2 | 1.00 | 108 |
| Arm 2 - Periodic Retrain | 1.610653 | 4.824570 | 2.334503 | 3 | 3 | 0 | 4 | 1.00 | 108 |
| Arm 3 - Feedback-Triggered ReAct | 1.533563 | 4.745950 | 2.554740 | 4 | 1 | 1 | 2 | 26.06 | 108 |
| Arm 4 - Feedback-Triggered No-Gate | 1.529448 | 4.630559 | 2.468103 | 4 | 3 | 1 | 15 | 26.06 | 108 |

Recovery is measured from the first 10-episode rolling MAE above baseline×1.2 to the first later value at or below baseline×1.1. N/A means no qualifying recovery was observed before replay ended.
