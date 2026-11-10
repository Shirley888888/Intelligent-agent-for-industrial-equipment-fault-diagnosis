@echo off
setlocal
cd /d %~dp0
python train_chronos2_ett.py --config configs/chronos2_ett_15min_loso_m1_to_m2.json --device-map cuda
if errorlevel 1 exit /b %errorlevel%
python train_chronos2_ett.py --config configs/chronos2_ett_15min_loso_m2_to_m1.json --device-map cuda
if errorlevel 1 exit /b %errorlevel%
python train_chronos2_ett.py --config configs/chronos2_ett_15min.json --device-map cuda
if errorlevel 1 exit /b %errorlevel%
echo 15-minute cross-station evaluation and pooled fine-tuning completed.
