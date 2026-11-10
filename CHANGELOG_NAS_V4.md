# NAS V4 change log

- Added registered innovation operators: trend/residual recombination, FFT low/high-frequency mixer, and cross-variable attention.
- Added optimizer, learning-rate, weight-decay, scheduler, and loss-weight search dimensions.
- Added PyTorch optimizer/scheduler builders for host training callbacks.
- Added architecture crossover between validated parent candidates.
- Added paired-seed parent references to ablation records and experiment-derived falsifiable structural hypotheses.
- Added next-generation seed population output and historical NAS topology reuse in the research cycle. Historical candidates are retrained on the current split before ranking.
- Kept declarative allowlists, parameter budgets, and deployment promotion disabled.
