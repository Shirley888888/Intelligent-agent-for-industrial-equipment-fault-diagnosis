@echo off
setlocal
cd /d %~dp0
python train_chronos2_ett.py --config configs/chronos2_ett_hourly.json --device-map cuda
if errorlevel 1 exit /b %errorlevel%
echo Hourly Chronos-2 run finished. Review models/chronos2_ett/hourly/manifest.json
