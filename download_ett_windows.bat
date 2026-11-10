@echo off
setlocal
cd /d %~dp0
python download_ett.py --output-dir data/ett
if errorlevel 1 exit /b %errorlevel%
echo ETT-small files downloaded and validated. See data/ett/source_manifest.json
