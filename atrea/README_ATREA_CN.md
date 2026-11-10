# ATREA functional integration update

The ATREA package now includes a trigger-gated orchestrator (`atrea.orchestrator.run_if_due`) that calls the research cycle only when performance, PSI, or schedule triggers fire. The cycle supports delta arXiv searches, validates experiment fact records, invokes an injected candidate gate, optionally invokes an explicitly enabled activation callback, writes academic-format reports, and induces auditable scoped pairwise patterns from repeated same-split experiments.

## Production integration contract

The host application must provide current matured forecast errors, reference/recent feature windows, training timestamps, and an `experiment_runner` that executes incumbent and registered candidates with identical chronological splits and preprocessing. The runner returns records with `dataset`, `split_hash`, `model_name`, `config`, `hyperparameters`, and `metrics` (including MAE to induce pairwise patterns). Each candidate must be marked `is_candidate`. A host `candidate_gate(candidate, records)` must enforce its frozen validation protocol, uncertainty/statistical policy, operational limits, and leakage controls. Activation is disabled by default; enabling it requires a host `activate_candidate(candidate, gate_result)` callback connected to the existing versioned release/rollback system. Research code alone must not publish a model.

Paper abstract keyword extraction is discovery assistance, not implementation or evidence. The default registry remains empty intentionally: each model needs a reviewed adapter, pinned dependency/weight revision, license and resource metadata. Chronos-2 can be registered lazily with `register_optional_chronos2`; no weights are downloaded on import.

## Synthetic end-to-end smoke demo

From the repository root, run `python -m atrea.cli --synthetic-demo --workspace work/atrea-demo.sqlite3`. It uses fabricated errors, features, paper metadata, and model scores. The report and experiment rows are explicitly labeled synthetic and must never be cited as benchmark evidence. For a real run, call `run_if_due` from the host service with the real data providers described above.

## Boundaries

The workflow is autonomous from trigger decision through experiment report when host callbacks are configured. Automatic activation is an opt-in policy decision and must be backed by the host's release gate. The research loop does not continuously reschedule itself; invoke `run_if_due` from the host scheduler after new feedback/data arrives.
