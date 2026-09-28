# Agent Test Cases

## Real ETTh1 scenario tests

1. `stable/` — stable operating condition
2. `slow_rise/` — slow temperature-rise condition
3. `fast_rise/` — fast temperature-rise condition

Each case directory contains `input.csv`, the 96-hour × 7-variable input.  <!-- modify:audit-output-path -->

Generated results are persisted under the project-level `output/` directory as `case_stable.json`, `case_slow_rise.json`, `case_fast_rise.json`, plus `all_case_results.json`.

Run all three in `stable -> slow_rise -> fast_rise` order and save an aggregate JSON:  <!-- modify:run-all-cases -->

```bat
python tests\run_agent_cases.py
```

## Risk boundary validation

<!-- modify:dynamic-risk -->
`test_risk_boundaries.py` verifies boundaries calculated from ETTh1 training-set quantiles rather than fixed 75/90°C values. The actual threshold values are printed at runtime.

Run:

```bat
python tests\test_risk_boundaries.py
```

These boundary tests are separate from the real ETTh1 scenario tests:
the scenario tests validate the end-to-end Agent workflow, while the
boundary tests validate the deterministic risk-rule implementation.
