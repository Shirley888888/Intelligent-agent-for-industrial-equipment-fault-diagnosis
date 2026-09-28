@echo off
chcp 65001 > nul
cd /d "C:\Users\limin\Desktop\我的\论文\4\ETTh1_Industrial_Agent_V8_Paper_Demo_modified\ETTh1_Industrial_Agent_V8_Paper_Demo"
set PYTHONIOENCODING=utf-8
echo Working directory: %CD%
echo.
echo Starting GPU training...
echo.
python train_gpu.py
echo.
echo Training complete. Exit code: %ERRORLEVEL%