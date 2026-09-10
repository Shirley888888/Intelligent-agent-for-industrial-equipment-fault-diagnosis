# V4 paper-readiness changes

This version is prepared from the uploaded V3 GPU project.

Changes:
1. LinearModel is now a strict linear mapping from flattened 96x7 history to 24 OT outputs.
2. Training model construction filters model-only hyperparameters so optimizer/training settings are not passed to model constructors.
3. Agent checkpoint loading uses the same model-only parameter filtering.
4. Checkpoints record stopped_epoch and parameter_count when the corresponding training code path is used.
5. Existing chronological split, train-only scaling, five-model design, and GPU training configuration are preserved.

Before using final numbers in a paper:
- run the complete formal experiment again after this structural change;
- verify all five checkpoints can be loaded by run_agent.py;
- record actual stopped epoch, best epoch, parameter count, and training time;
- report the exact metric definitions and risk thresholds.

## V4.1 hotfix

The model constructor filter is now model-specific. In particular, the strict Linear model receives no
dropout/training-only arguments, while MLP/CNN1D/LSTM/TCN receive only their declared architecture
hyperparameters. The Agent predictor uses the same mapping.
