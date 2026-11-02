@echo off
setlocal
cd /d %~dp0

echo [0/4] Checking Chronos, PEFT and CUDA runtime...
python -c "import torch, chronos, peft; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA is not available'"
if errorlevel 1 exit /b %errorlevel%

echo [1/4] Download and verify official ETT-small files...
python download_ett.py --output-dir data/ett
if errorlevel 1 exit /b %errorlevel%

echo [2/4] Run data and split preflight...
python train_chronos2_ett.py --config configs/chronos2_ett_hourly.json --preflight-only
if errorlevel 1 exit /b %errorlevel%
python train_chronos2_ett.py --config configs/chronos2_ett_15min.json --preflight-only
if errorlevel 1 exit /b %errorlevel%

echo [3/4] Run 20-step hourly LoRA smoke training...
python train_chronos2_ett.py --config configs/chronos2_ett_hourly.json --device-map cuda --num-steps 20 --batch-size 2 --gradient-accumulation-steps 2 --out-dir models/chronos2_smoke_hourly
if errorlevel 1 exit /b %errorlevel%

echo [4/4] Run 20-step 15-minute LoRA smoke training...
python train_chronos2_ett.py --config configs/chronos2_ett_15min.json --device-map cuda --num-steps 20 --batch-size 2 --gradient-accumulation-steps 2 --out-dir models/chronos2_smoke_15min
if errorlevel 1 exit /b %errorlevel%

echo Smoke runs complete. Inspect candidate_gate.json and memory usage before full training.
echo Smoke checkpoints are candidates only and are not auto-activated.
