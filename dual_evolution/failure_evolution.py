from __future__ import annotations

"""Failure-driven candidate experimentation with a hard production boundary.

This module coordinates experiments only. It never deploys or mutates the
production model; a PASS candidate is merely eligible for a separate shadow /
release workflow.
"""

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .common import canonical_json, require, sha256_json


@dataclass
class FailureEvolutionConfig:
    max_attempts_per_candidate_family: int = 3
    max_total_attempts: int = 6
    max_total_training_seconds: float = 3600.0
    retry_same_failure_signature: bool = False

    def validate(self) -> None:
        for key in ("max_attempts_per_candidate_family", "max_total_attempts"):
            value = getattr(self, key)
            require(type(value) is int and value >= 1, f"{key} must be a positive integer")
        require(self.max_total_training_seconds > 0,
                "max_total_training_seconds must be positive")
        require(type(self.retry_same_failure_signature) is bool,
                "retry_same_failure_signature must be boolean")


class FailureExperienceLedger:
    """Append-only, hash-chained experiment ledger (single-writer contract)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def read(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        previous = "0" * 64
        with self.path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Failure ledger JSON error at line {line_no}") from exc
                digest = row.pop("event_hash", None)
                if row.get("prev_hash") != previous or digest != sha256_json(row):
                    raise ValueError(f"Failure ledger hash-chain error at line {line_no}")
                row["event_hash"] = digest
                rows.append(row)
                previous = digest
        return rows

    def append(self, payload: dict[str, Any]) -> dict[str, Any]:
        rows = self.read()
        body = {"sequence": len(rows) + 1, "event_id": str(uuid.uuid4()),
                "created_at": self._now(),
                "prev_hash": rows[-1]["event_hash"] if rows else "0" * 64,
                "payload": payload}
        event = {**body, "event_hash": sha256_json(body)}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return event


def diagnose_candidate_failure(training: dict[str, Any] | None,
                                gate: dict[str, Any] | None,
                                error: str | None = None) -> dict[str, Any]:
    """Produce evidence-linked failure labels; do not let an LLM invent causes."""
    categories: list[str] = []
    evidence: list[str] = []
    if error:
        categories.append("training_or_runtime_error")
        evidence.append("training_exception")
    training = training or {}
    gate = gate or {}
    failures = list(gate.get("failures", []))
    insufficient = list(gate.get("insufficient_evidence", []))
    for item in failures:
        key = str(item).lower()
        if "mae" in key or "rmse" in key or "error" in key:
            category = "prediction_quality_regression"
        elif "stratum" in key or "fault_type" in key or "equipment" in key:
            category = "slice_or_robustness_regression"
        else:
            category = "gate_constraint_violation"
        if category not in categories:
            categories.append(category)
        evidence.append(str(item))
    if insufficient:
        categories.append("insufficient_validation_evidence")
        evidence.extend(str(x) for x in insufficient)
    metrics = training.get("metrics", {})
    if training.get("status") not in (None, "completed", "success"):
        categories.append("incomplete_training")
        evidence.append(f"training_status={training.get('status')}")
    if training.get("train_seconds") is not None:
        evidence.append(f"train_seconds={training['train_seconds']}")
    if not categories:
        categories.append("unknown_or_unclassified")
    return {"categories": sorted(set(categories)), "evidence": evidence,
            "observed_metrics": metrics, "gate_status": gate.get("status", "NOT_RUN"),
            "root_cause_confidence": "evidence_only_not_causal_proof"}


def _failure_signature(diagnosis: dict[str, Any], candidate: dict[str, Any]) -> str:
    # Repeated failure is scoped to the same architecture hypothesis. A failed
    # Transformer must not prevent trying a materially different CNN-Transformer.
    strategy = candidate.get("architecture", candidate.get("model_name",
                         candidate.get("strategy", sha256_json(candidate))))
    return sha256_json({"strategy": strategy, "categories": diagnosis["categories"],
                        "evidence": sorted(diagnosis["evidence"])})


class FailureDrivenEvolution:
    """Run bounded candidate experiments and feed failures into the next proposal.

    Callbacks:
      train_and_evaluate(candidate, attempt) -> training result dict
      gate_evaluate(candidate, training_result) -> gate result dict
      propose_next(previous_candidate, diagnosis, history) -> next candidate | None

    A gate PASS returns ELIGIBLE_FOR_SHADOW, never PRODUCTION or ACTIVATED.
    """

    def __init__(self, ledger: FailureExperienceLedger,
                 config: FailureEvolutionConfig | None = None):
        self.ledger = ledger
        self.config = config or FailureEvolutionConfig()
        self.config.validate()

    def run(self, initial_candidate: dict[str, Any], *,
            train_and_evaluate: Callable[[dict[str, Any], int], dict[str, Any]],
            gate_evaluate: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
            propose_next: Callable[[dict[str, Any], dict[str, Any], list[dict[str, Any]]],
                                   dict[str, Any] | None] | None = None,
            candidate_family: str = "default") -> dict[str, Any]:
        require(isinstance(initial_candidate, dict) and bool(initial_candidate),
                "initial_candidate must be a nonempty object")
        family_rows = [r["payload"] for r in self.ledger.read()
                       if r.get("payload", {}).get("candidate_family") == candidate_family]
        history = list(family_rows)
        seen_configs = {row.get("candidate_hash") for row in history}
        failure_signatures = {row.get("failure_signature") for row in history
                              if row.get("failure_signature")}
        candidate = dict(initial_candidate)
        total_seconds = 0.0
        attempts: list[dict[str, Any]] = []
        family_remaining = self.config.max_attempts_per_candidate_family - len(history)
        total_remaining = self.config.max_total_attempts - len(history)
        max_attempts = max(0, min(family_remaining, total_remaining))
        if max_attempts == 0:
            return {"status": "STOPPED_EXPERIMENT_BUDGET", "attempt_count": 0,
                    "total_training_seconds": 0.0, "attempts": [],
                    "eligible_candidate": None, "production_model_changed": False,
                    "next_stage": "research_or_wait_for_new_evidence",
                    "config": asdict(self.config),
                    "ledger_integrity": {"valid": True, "event_count": len(self.ledger.read())}}
        final_status = "BUDGET_EXHAUSTED"

        for attempt in range(1, max_attempts + 1):
            candidate_hash = sha256_json(candidate)
            if candidate_hash in seen_configs:
                final_status = "STOPPED_DUPLICATE_CONFIGURATION"
                break
            seen_configs.add(candidate_hash)
            training: dict[str, Any] = {}
            gate: dict[str, Any] = {"status": "NOT_RUN"}
            error = None
            try:
                training = train_and_evaluate(candidate, attempt) or {}
                elapsed = float(training.get("train_seconds", 0.0) or 0.0)
                if elapsed < 0:
                    raise ValueError("train_seconds cannot be negative")
                total_seconds += elapsed
                if total_seconds > self.config.max_total_training_seconds:
                    error = "training_budget_exceeded"
                elif training.get("status", "completed") not in ("completed", "success"):
                    error = f"training_status:{training.get('status')}"
                else:
                    gate = gate_evaluate(candidate, training) or {"status": "REVIEW_REQUIRED"}
            except Exception as exc:  # failure is recorded, not silently swallowed
                error = f"{type(exc).__name__}: {exc}"

            passed = gate.get("status") == "PASS" and not error
            diagnosis = {"categories": [], "evidence": [], "observed_metrics": {},
                         "gate_status": gate.get("status", "NOT_RUN"),
                         "root_cause_confidence": "not_applicable"} if passed else \
                        diagnose_candidate_failure(training, gate, error)
            signature = None if passed else _failure_signature(diagnosis, candidate)
            row = {
                "candidate_family": candidate_family,
                "candidate_hash": candidate_hash,
                "candidate": candidate,
                "attempt": len(history) + len(attempts) + 1,
                "training": training,
                "gate": gate,
                "error": error,
                "diagnosis": diagnosis,
                "failure_signature": signature,
                "outcome": "PASS" if passed else ("REVIEW_REQUIRED" if gate.get("status") == "REVIEW_REQUIRED" else "REJECT"),
                "production_model_changed": False,
                "created_at": FailureExperienceLedger._now(),
            }
            event = self.ledger.append(row)
            row["event_hash"] = event["event_hash"]
            attempts.append(row)
            history.append(row)

            if passed:
                final_status = "ELIGIBLE_FOR_SHADOW"
                break
            if total_seconds >= self.config.max_total_training_seconds:
                final_status = "STOPPED_TRAINING_BUDGET"
                break
            if attempt >= max_attempts or propose_next is None:
                final_status = "STOPPED_RETRY_LIMIT"
                break
            if signature in failure_signatures and not self.config.retry_same_failure_signature:
                final_status = "STOPPED_REPEATED_FAILURE_SIGNATURE"
                break
            failure_signatures.add(signature)
            next_candidate = propose_next(candidate, diagnosis, list(history))
            if next_candidate is None:
                final_status = "NO_FURTHER_CANDIDATE"
                break
            if not isinstance(next_candidate, dict) or not next_candidate:
                raise ValueError("propose_next must return a nonempty candidate object or None")
            candidate = dict(next_candidate)

        return {
            "status": final_status,
            "attempt_count": len(attempts),
            "total_training_seconds": total_seconds,
            "attempts": attempts,
            "eligible_candidate": attempts[-1]["candidate"] if final_status == "ELIGIBLE_FOR_SHADOW" else None,
            "production_model_changed": False,
            "next_stage": "independent_shadow_validation_required" if final_status == "ELIGIBLE_FOR_SHADOW" else "research_or_wait_for_new_evidence",
            "config": asdict(self.config),
            "ledger_integrity": {"valid": True, "event_count": len(self.ledger.read())},
        }
