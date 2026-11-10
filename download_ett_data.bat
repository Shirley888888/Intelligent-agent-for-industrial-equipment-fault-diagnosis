@echo off
setlocal
python download_ett.py --output-dir data/ett
if errorlevel 1 exit /b %errorlevel%
echo ETT-small data download and validation completed.
endlocal
