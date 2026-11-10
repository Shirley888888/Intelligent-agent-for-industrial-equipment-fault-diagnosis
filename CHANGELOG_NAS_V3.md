# ATREA NAS V3 Change Log

## Added
- `atrea/nas.py`: bounded evolutionary topology search, mutation, loss objectives, ablation generation and ranking.
- `NAS_Composed` in `models.py`: multi-branch compositional network with configurable block sequences.
- Attention placement: input, after each block, branch end, fusion, or disabled.
- Fusion strategies: concatenation, mean and learned gated fusion.
- Four loss objectives: MAE, Huber, MAE+RMSE and horizon-weighted MAE.
- `run_nas_search`: multi-generation search with cross-seed mean ranking and optional leave-one-component-out ablation.
- Optional integration in `run_research_cycle` through injected `nas_train_evaluate`, `nas_ablation_evaluate` and mandatory `nas_split_hash`.
- Research planner now supplies an initial NAS seed population and records search dimensions.
- `NAS_GUIDE_CN.md` documents host integration and evaluation constraints.

## Safety / reproducibility
- Search specs are declarative and allowlisted; no generated Python is executed.
- Topology, width, branch count, block count and parameter budgets are bounded.
- Candidate ranking uses validation metrics aggregated across seeds.
- NAS never authorizes deployment by itself (`promotion_allowed=False`); existing release gates remain required.
- Final untouched test set must not be used during architecture selection.

## Verification
- `tests_agent`: 46 tests passed.
- Full project unittest discovery: 29 tests passed, 1 CUDA-only test skipped (CUDA unavailable).
- Additional smoke test: all 15 combinations of 5 attention placements x 3 fusion strategies passed forward-shape checks.
- `compileall` passed for modified modules.
- No full real ETTh1/industrial GPU training was performed in this build environment; real performance depends on the host training callback and user's dataset/runtime.
