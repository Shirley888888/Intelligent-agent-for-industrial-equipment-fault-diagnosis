# Industrial Diagnostic Agent: Four Iterations and Verification

## Scope and invariant

This is a first runnable Agent runtime wired to the V0 dual-evolution infrastructure. It consumes a structured prediction from the existing independent LSTM/TCN model, uses an optional LLM planner to select allowlisted read-only tools, searches only approved/effective knowledge documents, produces a traceable report, and records an Episode. The Agent never predicts or changes the fault class. It does not operate equipment or execute repair commands.

The bundled offline planner makes the sample and CI tests reproducible without network access or API keys. A real OpenAI-compatible Chat Completions endpoint can be used by setting `OPENAI_API_KEY`, `OPENAI_MODEL`, and optionally `OPENAI_BASE_URL`. A deployed LLM is not bundled, and no external provider call is represented as tested in this package.

## Iteration 1 — Basic end-to-end diagnostic Agent

Implemented: typed input contract; model prediction provenance; tool-loop abstraction; KnowledgeCatalog retrieval; structured report; EpisodeStore write; optional OpenAI-compatible provider; deterministic offline provider.

Solved failure: before this, the V0 gate and ledger existed but nothing called them from a diagnostic Agent runtime.

Validation: offline full run, output provenance, Episode persistence, bad timestamp/provenance and non-finite sensor input.

## Iteration 2 — Tool reliability and least privilege

Implemented: explicit tool allowlist and JSON argument checks; equipment scope is taken from trusted context rather than LLM arguments; no shell/network/SQL/actuator tools; per-tool audit log; tool errors are isolated and preserved.

Solved failure: prompt injection or a malformed planner call cannot expand the Agent's capabilities or search another equipment's knowledge scope.

Validation: unknown tool rejection, unexpected argument rejection, scope override test, numeric sensor summaries, malformed tool input.

## Iteration 3 — Runtime safety and evidence constraints

Implemented: deterministic OOD/drift/confidence guard; fail-closed behavior for missing calibration or model signals; human-review ticket queue; no definitive decision when guard is conservative; recommendation source IDs must match actual retrieved knowledge hits; LLM-generated fault labels are ignored.

Solved failure: the Agent cannot convert an uncertain model output into a confident diagnosis or create an unsupported maintenance instruction.

Validation: critical OOD, uncalibrated high confidence, missing calibration, missing model label, untrusted sensor strings and hallucinated citation tests.

## Iteration 4 — Feedback, regression evaluation and governance

Implemented: Episode hash-chain integration; human feedback submission with source reference and label-state transitions; frozen Agent gate compatibility; tamper test; versioned prompt/policy/tool/KB/risk metadata; synthetic CLI demo.

Solved failure: Agent failures and human corrections no longer disappear in logs; their evidence can become reviewed Episodes and replay cases. The evaluator is external and rule-based, not an LLM grading itself.

Validation: Episode review, source-reference enforcement, Agent gate pass/reject, ledger integrity/tamper detection.

## Test summary

Run:

```bash
python -m unittest discover -s tests_agent -v
python -m unittest discover -s tests_dual -v
python -O -m unittest -v test_loop test_torch_version_checks
python -m diagnostic_agent.cli demo --workspace agent_workspace_demo
```

The first three commands check Agent tests, the dual-evolution modules, and the pre-existing model-loop regression suite. CUDA-specific checks may skip when CUDA is unavailable.

## Input contract

The Agent accepts JSON with `equipment_id`, ISO-8601 `prediction_time`, `operating_condition`, `model.model_version`, `model.activation_id`, and a `model_prediction` object supplied by the independent model. The prediction should include `fault_type`, `confidence`, `confidence_calibrated`, `ood_score`, and `drift_score`. Missing runtime safety signals or calibration force human review. `sensor_readings` is an optional map of finite numeric values or numeric arrays.

## Run with offline planner

```bash
python -m diagnostic_agent.cli run --input examples/agent/sample_context.json --workspace agent_workspace
```

The sample has no production calibration ID by default and therefore intentionally fails closed into human review. That is expected until a site-specific calibration identifier and thresholds are configured.

## Run with an OpenAI-compatible planner

```bash
# Configure outside the source tree; never commit secrets.
export OPENAI_API_KEY="..."
export OPENAI_MODEL="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"  # optional
export AGENT_CALIBRATION_ID="validated-calibration-id"
python -m diagnostic_agent.cli run --input path/to/model_prediction.json --workspace agent_workspace --provider openai
```

The LLM is only a report/evidence assistant. The fault label is always copied from `model_prediction.fault_type` in the upstream model result.

## Production gaps deliberately not claimed as complete

- The uploaded archive does not contain a real industrial fault-model serving API, actual model output calibration artifact, OEM manuals/SOP corpus, maintenance ticket connector, identity/authorization service, or production LLM endpoint credentials.
- Knowledge retrieval currently uses the project's approved-document TF-IDF baseline. Embedding retrieval/reranking is not configured because there is no real corpus to evaluate.
- EpisodeStore and tool logs are local single-writer files. Production needs transactional/concurrent storage, access control, retention policies and data governance.
- Runtime thresholds in examples are placeholders; site-specific calibration and fail-safe review by engineers are required.
- This is an advisory diagnosis/reporting Agent; no actuator or automatic maintenance action is exposed.

## Final verification run (2026-11-02)

| Suite | Tests | Result |
|---|---:|---|
| Agent runtime (`tests_agent`) | 29 | 29 passed |
| Dual-evolution V0 (`tests_dual`) | 12 | 12 passed |
| Original model loop + Torch version checks | 29 | 28 passed, 1 CUDA-dependent test skipped because CUDA is unavailable in this runtime |
| **Total** | **70** | **69 passed, 1 skipped, 0 failed** |

Additional checks: `py_compile` passed. The CLI demo was executed in two modes: (1) without a calibration ID, it correctly requests human review, stores the Episode and creates a review ticket; (2) with a clearly demo-only calibration ID, it returns a provisional model result, carries one synthetic approved-document citation and stores the Episode. Neither run is real equipment validation. OpenAI-compatible request/response handling and timeout behavior were tested with mocked HTTP responses; no live LLM API request was made.

Detailed logs and machine-readable summary are in `verification_agent/`. `AGENT_FILE_MANIFEST.sha256` lists hashes for the new Agent source, tests and sample input.
