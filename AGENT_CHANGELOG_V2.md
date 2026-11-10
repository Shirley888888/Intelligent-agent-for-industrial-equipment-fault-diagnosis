# Agent Change Log V2

- Fixed missing NumPy import in the autonomous research cycle.
- Added two-source paper discovery (arXiv + Semantic Scholar), recency/relevance candidate ranking, and provider error reporting.
- Added auditable ReAct event trace with event hashes; logs action summaries rather than private chain-of-thought.
- Added memory-scope migrations and isolated A1/A2/A3 experience; A2 paper/experiment outputs are artifacts only and do not enter reusable memory.
- Tightened experience induction to paired seeds on the same dataset and split.
- Added residual error diagnostics and per-arm Markdown reports.
- Made scheduler failures durable and retryable, with one successful run per China-local calendar day.
- Delayed candidate activation until experiment rows and reports are persisted; missing release checks block promotion.
- Added Chronos2ExperimentRunner to call the repository's existing real Chronos-2 fine-tuning script with validated config overrides.
- Added daily job template and example Windows launcher/config.
- Added tests for memory isolation, cross-run A3 retrieval, scheduler retry/idempotency, residual metrics, ReAct trace, and Chronos config validation.

## Explicit limitations

- A planner factory backed by the user's configured LLM service is still required for paper+memory-driven plan generation.
- The included Chronos-2 adapter does not change the foundation model architecture; A2/A3 require a separate multi-architecture training adapter.
- Automatic model activation remains disabled until the host connects a production registry, shadow evaluation, rollback and release approvals.
- Passing tests validates software logic, not a new GPU training run.
