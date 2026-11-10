@echo off
setlocal
cd /d %~dp0
python train_chronos2_ett.py --config configs/chronos2_ett_hourly_loso_h1_to_h2.json --device-map cuda
if errorlevel 1 exit /b %errorlevel%
python train_chronos2_ett.py --config configs/chronos2_ett_hourly_loso_h2_to_h1.json --device-map cuda
if errorlevel 1 exit /b %errorlevel%
python train_chronos2_ett.py --config configs/chronos2_ett_hourly.json --device-map cuda
if errorlevel 1 exit /b %errorlevel%
echo Hourly cross-station evaluation and pooled fine-tuning completed.
