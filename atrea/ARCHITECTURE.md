# ATREA architecture and autonomous cycle

```mermaid
flowchart TB
  S[Scheduler / Agent Orchestrator] --> T[Training Trigger]
  S --> P[Paper Discovery]
  T --> S
  P --> H[Hypothesis Builder]
  E[(SQLite Experience Store)] --> H
  H --> R[Model Registry]
  R --> X[Controlled Experiment Runner]
  X --> V[Evaluation + Statistical Gate]
  V -->|accept candidate| G[Approval / Shadow Deployment]
  V -->|reject candidate| E
  G --> M[Online Monitor + Rollback]
  M --> T
  X --> F[Fact Records]
  F --> I[Pattern Induction]
  I --> E
  X --> A[Academic Abstract Generator]
  A --> E
```

## Full autonomous cycle

```mermaid
sequenceDiagram
  participant C as Scheduler
  participant T as Drift Monitor
  participant P as Paper Search
  participant L as Research Planner (LLM)
  participant R as Model Registry
  participant X as Experiment Runner
  participant G as Statistical Gate
  participant D as Experience DB
  C->>T: inspect mature feedback + data window
  T-->>C: retrain trigger + evidence
  C->>P: search new papers since last scan
  P-->>L: abstracts, metadata, methods, code links
  D-->>L: prior facts and induced patterns
  L->>L: formulate falsifiable hypothesis + protocol
  L->>R: request candidate adapters
  R-->>X: instantiate candidate and incumbent
  X->>X: same chronological splits and preprocessing
  X-->>G: paired predictions on frozen validation/test
  G-->>C: pass / reject / insufficient evidence
  X->>D: append immutable experiment facts
  X->>D: store Abstract, Methods, Results, Discussion, Conclusion
  D->>L: update evidence for next research cycle
```

## Required controls
- Separate train/validation/test chronologically. Tune only on validation; use the final test once per model-selection protocol.
- For multi-equipment transfer, add leave-one-equipment-out evaluation; ETTh1/ETTh2 are not independent industrial equipment fleets.
- Store split hash, raw data hash, preprocessing version, random seed, source commit and dependency lock.
- Bootstrap paired errors by forecast origin (not by individual overlapping horizon points) to avoid overstating effective sample size. Use block bootstrap where temporal dependence remains.
- Statistical significance is necessary but not sufficient: enforce absolute error limits, tail-risk limits and critical operating-condition/fault strata.
- Candidate code and weights run in an isolated environment. Paper-derived code is untrusted; never execute arbitrary install scripts or repository code automatically.
- Promotion requires an explicit policy gate, shadow evaluation, monitoring and rollback. Research agents may propose; they must not silently deploy.
