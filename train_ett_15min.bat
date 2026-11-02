@echo off
setlocal
python train_ett_corpus.py --data-dir data/ett --frequency 15min --model both --out-dir models/ett_corpus --device cuda:0 --epochs 10 --patience 3
if errorlevel 1 exit /b %errorlevel%
echo 15-minute ETTm1+ETTm2 LSTM/TCN training completed.
endlocal
