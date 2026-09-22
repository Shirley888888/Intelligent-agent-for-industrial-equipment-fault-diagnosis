# V7 Experimental Validation Report

## Scope
V7 organizes the supplied V6 Agent into paper-oriented experimental artifacts. No model
retraining is performed.

Setup: ETTh1, 7 input variables, 96-hour input window, 24-hour OT forecast horizon,
chronological 70/15/15 split.

## Model comparison
The packaged `outputs/metrics.csv` is copied unchanged to `results/model_comparison.csv`.
Best MAE in this packaged result: **LSTM (1.364470)**.
Best MSE in this packaged result: **LSTM (3.489796)**.

## Real ETTh1 case studies
Three inherited real-data windows are reported: stable, slow_rise, and fast_rise.
Detailed results are in `results/case_study_results.csv`.

The case labels describe historical input behavior; they do not force the forecast to preserve
the same ordering.

## Diagnosis decision validation
`results/diagnosis_decision_validation.csv` contains synthetic decision-layer checks.
These are explicitly separated from measured ETTh1 results and should not be reported as
ground-truth fault labels.

## Figures
- model_comparison_mae.png
- model_comparison_mse.png
- case_stable_forecast.png
- case_slow_rise_forecast.png
- case_fast_rise_forecast.png
- case_study_summary.png

## Limitation
ETTh1 provides oil-temperature/load time series rather than ground-truth transformer fault
labels. Therefore NORMAL/WARNING/ANOMALY is a rule-based thermal decision aid, not a validated
fault classifier. Industrial deployment would require site-specific thresholds, fault labels,
and engineering validation.
