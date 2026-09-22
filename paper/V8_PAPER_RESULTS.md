# V8 Paper & Demo Edition — Results Narrative

## Experimental setup
The project uses the ETTh1 benchmark with seven variables (`HUFL, HULL, MUFL, MULL, LUFL,
LULL, OT`). The model input is 96 hourly observations and the prediction horizon is 24 hours.
The target is OT. The chronological split is 70%/15%/15%, with the scaler fitted only on the
training portion.

## Model comparison
Among the packaged model-comparison results, LSTM achieves the lowest MAE
(1.364470) and MSE (3.489796). Relative to the Persistence baseline
(MAE 1.459667, MSE 3.994103), LSTM reduces MAE by 6.52% and MSE
by 12.63% in this test result.

The TCN has the lowest PeakError (1.300669), while CNN1D has the lowest
MaxRiseRateError (2.469316). These metric-specific strengths
support retaining the five structurally different architectures in the comparison while
using LSTM as the current Agent forecasting model.

## Agent demonstration
The verified default demonstration produces a 24-hour forecast with mean 9.0445 °C and maximum
10.7793 °C. The forecast maximum positive slope is 0.7185 °C/h. The historical anomaly tool
returns NORMAL with score 0.0, and the diagnosis engine returns NORMAL.

## Case studies
Three real ETTh1 windows are included: stable, slow_rise, and fast_rise. The case labels describe
the historical input behavior; they do not imply that the forecast must preserve the same ordering.

## Interpretation and limitation
The diagnosis layer is an interpretable rule-based thermal decision aid combining historical
temperature behavior and forecast risk. ETTh1 does not provide ground-truth transformer fault
type labels. Therefore, the current experiments do not establish fault-classification accuracy.
For real industrial deployment, thresholds and diagnostic conclusions require site-specific
engineering validation and labeled fault data.

## Recommended paper claim
A defensible claim is:
"An LSTM-based oil-temperature forecasting and interpretable thermal anomaly/risk diagnosis
agent was developed and evaluated on ETTh1."

Avoid claiming that ETTh1 results validate specific transformer fault types.
