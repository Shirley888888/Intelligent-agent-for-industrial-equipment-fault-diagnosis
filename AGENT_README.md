
# Complete Agent Addendum

This package completes the four Agent tasks in the assignment:
1. Model assets: `agent/models/best_model.pt`, `scaler.pkl`, `config.yaml`.
2. Predictor tool: validated 96h x 7 variables -> 24h OT.
3. Risk tool: `mean`, `max`, `rise`, `max_slope`, `risk_level`.
4. Three saved real-ETTh1 test cases: `stable`, `slow_rise`, `fast_rise`, each with `input.csv` and generated `prediction.json`.

## Run the final LSTM Agent
```bash
python run_agent.py
```
The default input uses the last 96 rows of `data/ETTh1.csv`.

## Run the three test cases
```bash
python tests/run_agent_cases.py
```

## Run one case
```bash
python run_agent.py --case stable
python run_agent.py --case slow_rise
python run_agent.py --case fast_rise
```

The three cases are real windows extracted from ETTh1, selected by the temperature behavior in their final 24 hours. They are demonstration scenarios, not synthetic data.

Risk thresholds are configurable in `agent/models/config.yaml`. The current thresholds are 75 C (medium) and 90 C (high), inherited from the existing project configuration; they are not claimed here as universal industry standards.
