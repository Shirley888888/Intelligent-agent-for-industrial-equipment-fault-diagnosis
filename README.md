# ETTh1 Industrial Oil-Temperature Forecasting Agent — V4

# ETTh1 Industrial Oil-Temperature Agent — v3 Full GPU

This is the complete runnable v3 experiment project for the official ETTh1 dataset.

## What it runs
- Input: 96 hourly steps × 7 variables
- Target: next 24 hourly OT values
- Chronological split: 70% / 15% / 15% **before** windowing
- Windows never cross split boundaries
- StandardScaler fitted only on training split
- Persistence baseline
- Exactly five deep architectures: Linear, MLP, CNN1D, LSTM, TCN
- CUDA auto-detection, AMP, TF32, gradient clipping, early stopping
- Best checkpoint with model/scaler/config metadata
- TensorBoard + CSV + JSON + XLSX experiment records
- Loss curves and forecast plots
- Agent prediction + LOW/MEDIUM/HIGH risk assessment

## Windows / Conda
```bat
conda activate etth1_gpu
cd /d "C:\Users\limin\Desktop\我的\论文\4\ETTh1_Industrial_Agent_v3_GPU"
python -m pip install -r requirements.txt
```
If PyTorch is already installed for RTX 5060, keep that CUDA-enabled installation. Do not replace it with a CPU-only wheel.

## Start full experiment
```bat
python train_gpu.py --device cuda:0
```

For a quick smoke test:
```bat
python train_gpu.py --device cuda:0 --epochs 2 --batch-size 128
```

Train selected models:
```bat
python train_gpu.py --device cuda:0 --models LSTM TCN
```

## TensorBoard
After training:
```bat
tensorboard --logdir outputs/tensorboard
```
Then open the local address printed by TensorBoard.

## Agent
The best checkpoint can be used by:
```bat
python run_agent.py --checkpoint outputs/checkpoints/TCN_best.pt
```

`history` must be 96 × 7 in the order:
`HUFL,HULL,MUFL,MULL,LUFL,LULL,OT`.

## Outputs
- `outputs/metrics.csv`
- `outputs/metrics.json`
- `outputs/metrics.xlsx`
- `outputs/run_config.json`
- `outputs/checkpoints/*_best.pt`
- `outputs/plots/*_loss.png`
- `outputs/plots/*_forecast.png`
- `outputs/tensorboard/`

The metrics are computed after inverse-transforming OT back to its original temperature scale.


## V4 audit status
- Linear model corrected to a strict linear baseline.
- Agent checkpoint loading uses filtered model hyperparameters.
- Re-run formal experiments before treating metrics as final paper results.
