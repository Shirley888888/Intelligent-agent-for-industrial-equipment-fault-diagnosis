# NAS V5 Changelog

- Added learnable amplitude-phase frequency operator.
- Added weakness-driven operator proposal generator.
- Added multi-objective Pareto front over available MAE, RMSE, parameter count and training time.
- Evolutionary survivor selection now preserves Pareto diversity before filling by primary metric.
- ETTh1 training callback records elapsed training seconds.
- Research cycle passes diagnostic weakness to NAS and persists Pareto candidate summaries.
- Promotion remains disabled pending untouched-test and release gates.
