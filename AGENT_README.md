
# Complete Agent Addendum

This package completes the four Agent tasks in the assignment:
1. Model assets: `agent/models/best_model.pt`, `scaler.pkl`, `config.yaml`.
2. Predictor tool: validated 96h x 7 variables -> 24h OT.
3. Risk tool: `mean`, `max`, `rise`, `max_slope`, `risk_level`.
4. Three saved real-ETTh1 test inputs: `stable`, `slow_rise`, `fast_rise`. Running the case runner writes the complete Agent outputs to `output/case_<name>.json`.  <!-- modify:audit-output-path -->

## Run the final LSTM Agent
```bash
python run_agent.py
```
<!-- modify:run-all-cases -->
The default runs all three cases and saves complete results. To forecast the last 96 rows of ETTh1, use `python run_agent.py --input data/ETTh1.csv`.

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

<!-- modify:dynamic-risk -->
Risk thresholds are calculated dynamically from the chronological ETTh1 training split. The Agent uses training-window quantiles for forecast peak temperature, 24-hour temperature rise, and maximum one-hour slope; quantile levels are configured in `agent/models/config.yaml`.

<!-- modify:model-unification -->
The final Agent model is **LSTM** everywhere: `agent/models/config.yaml` declares `LSTM`, `agent/models/best_model.pt` contains `model_name=LSTM`, and the run commands/documentation use that same checkpoint.

<!-- modify:run-all-cases -->
`python tests/run_agent_cases.py` runs `stable`, `slow_rise`, and `fast_rise` in that order, writes `output/case_stable.json`, `output/case_slow_rise.json`, `output/case_fast_rise.json`, and creates the aggregate `output/all_case_results.json`.  <!-- modify:audit-output-path -->
