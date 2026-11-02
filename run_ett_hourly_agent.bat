@echo off
setlocal
python -m diagnostic_agent.cli run-ett-corpus --model-dir models/ett_corpus/hourly/LSTM --csv data/ett/ETTh2.csv --equipment-id ETT-STATION-2 --workspace agent_workspace_oil_temp --device cuda:0
endlocal
