@echo off
setlocal
cd /d %~dp0
python train_chronos2_ett.py --config configs/chronos2_ett_15min.json --device-map cuda
if errorlevel 1 exit /b %errorlevel%
echo 15-minute Chronos-2 run finished. Review models/chronos2_ett/15min/manifest.json
