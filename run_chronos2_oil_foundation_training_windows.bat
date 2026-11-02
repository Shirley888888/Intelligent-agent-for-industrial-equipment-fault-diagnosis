@echo off
setlocal
cd /d %~dp0

echo [0/4] Checking Chronos, PEFT and CUDA runtime...
python -c "import torch, chronos, peft; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA is not available'"
if errorlevel 1 exit /b %errorlevel%

echo [1/4] Download and verify the four official ETT-small transformer oil-temperature CSVs...
python download_ett.py --output-dir data/ett
if errorlevel 1 exit /b %errorlevel%

echo [2/4] Preflight the hourly and 15-minute datasets before loading model weights...
python train_chronos2_ett.py --config configs/chronos2_ett_hourly.json --preflight-only
if errorlevel 1 exit /b %errorlevel%
python train_chronos2_ett.py --config configs/chronos2_ett_15min.json --preflight-only
if errorlevel 1 exit /b %errorlevel%

echo [3/4] LoRA fine-tune Chronos-2 on hourly and 15-minute oil-temperature data...
python train_chronos2_ett.py --config configs/chronos2_ett_hourly.json --device-map cuda
if errorlevel 1 exit /b %errorlevel%
python train_chronos2_ett.py --config configs/chronos2_ett_15min.json --device-map cuda
if errorlevel 1 exit /b %errorlevel%

echo [4/4] Training runs finished. Review candidate_gate.json and final test metrics under models\chronos2_ett\.
echo Passing the candidate gate does NOT automatically activate a production model.
