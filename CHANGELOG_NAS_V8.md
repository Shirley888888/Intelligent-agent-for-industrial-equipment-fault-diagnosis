# NAS V8 Changelog

- Added bounded `darts_operator` mixed-operation nodes with trainable softmax architecture logits.
- Added `extract_darts_genotype` to export the highest-weight primitive as a discrete architecture.
- Added safe symbolic residual regression with a fixed expression grammar, chronological train/validation split, greedy forward selection, and ridge fitting.
- Added DARTS candidate injection to error-driven operator innovation proposals.
- Updated NAS search-mode identifier to `hypothesis_driven_differentiable_symbolic_nas_v8`.
- Added tests for architecture-logit gradients, genotype export, and symbolic trend recovery.

## Limitations

This release implements DARTS-style joint-gradient mixed operations, not the original bilevel DARTS optimization protocol. Symbolic regression currently returns a research artifact and does not automatically alter the deployed predictor.
