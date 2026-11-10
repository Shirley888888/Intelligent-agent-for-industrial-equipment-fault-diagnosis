# Change Log — Failure-Driven Evolution

## Added
- Added `dual_evolution/failure_evolution.py` for bounded autonomous retries after candidate-model gate rejection.
- Added append-only, SHA-256 hash-chained experiment memory with tamper detection.
- Added evidence-based failure labels for prediction regression, slice robustness regression, insufficient evidence, training failure, and runtime errors.
- Added duplicate-candidate protection, per-family/global attempt limits, and training-time budgets.
- Added a hard production boundary: passing candidates are only `ELIGIBLE_FOR_SHADOW`; this loop never activates or replaces the production model.
- Added `configs/failure_evolution.json`, Chinese integration guide, and six focused unit tests.

## Verification
- Focused failure-evolution tests: 6 passed.
- Existing `tests_agent`: 62 passed.
- `tests_dual`: 18 passed (including the 6 new tests).
- These are software-behavior tests, not evidence of forecasting accuracy improvement.
