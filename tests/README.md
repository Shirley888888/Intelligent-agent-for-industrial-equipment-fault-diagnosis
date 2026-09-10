# Agent Test Cases

## Real ETTh1 scenario tests

1. `stable/` — stable operating condition
2. `slow_rise/` — slow temperature-rise condition
3. `fast_rise/` — fast temperature-rise condition

Each case contains:
- `input.csv`: the 96-hour × 7-variable input
- `prediction.json`: the 24-hour LSTM forecast and risk analysis

Run all three:

```bat
python tests\run_agent_cases.py
```

## Risk boundary validation

`test_risk_boundaries.py` verifies the configurable temperature thresholds
without retraining or modifying the forecasting model:

- 74.9°C -> LOW
- 75.0°C -> MEDIUM
- 89.9°C -> MEDIUM
- 90.0°C -> HIGH

Run:

```bat
python tests\test_risk_boundaries.py
```

These boundary tests are separate from the real ETTh1 scenario tests:
the scenario tests validate the end-to-end Agent workflow, while the
boundary tests validate the deterministic risk-rule implementation.
