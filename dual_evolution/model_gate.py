from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from .common import require


@dataclass
class ModelGateConfig:
    """Starting profile only; final thresholds must be calibrated per device/task."""
    mae_max_degradation: float = 0.02
    p95_max_degradation: float = 0.05
    p99_max_degradation: float = 0.05
    max_error_max_degradation: float = 0.10
    stratum_mae_max_degradation: float = 0.10
    stratum_tail_max_degradation: float = 0.10
    p95_min_samples: int = 20
    p99_min_samples: int = 100
    min_stratum_samples: int = 20
    min_critical_stratum_samples: int = 5
    critical_fault_types: tuple[str, ...] = ()
    absolute_max_error_limit: float | None = None
    zero_baseline_epsilon: float = 1e-8

    def validate(self) -> None:
        for name in ("mae_max_degradation", "p95_max_degradation", "p99_max_degradation",
                     "max_error_max_degradation", "stratum_mae_max_degradation",
                     "stratum_tail_max_degradation", "zero_baseline_epsilon"):
            value = getattr(self, name)
            require(np.isfinite(value) and value >= 0, f"{name} must be finite and nonnegative")
        for name in ("p95_min_samples", "p99_min_samples", "min_stratum_samples",
                     "min_critical_stratum_samples"):
            require(type(getattr(self, name)) is int and getattr(self, name) >= 1,
                    f"{name} must be a positive integer")
        if self.absolute_max_error_limit is not None:
            require(np.isfinite(self.absolute_max_error_limit) and self.absolute_max_error_limit > 0,
                    "absolute_max_error_limit must be finite and positive")


def regression_metrics(y_true: Any, y_pred: Any, *, p95_min_samples: int = 20,
                       p99_min_samples: int = 100) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    require(truth.size == pred.size and truth.size > 0, "y_true/y_pred must have equal nonzero length")
    require(np.isfinite(truth).all() and np.isfinite(pred).all(), "Predictions and labels must be finite")
    errors = np.abs(truth - pred)
    n = int(errors.size)
    return {
        "n": n,
        "mae": float(np.mean(errors)),
        "p95_abs_error": float(np.quantile(errors, 0.95)) if n >= p95_min_samples else None,
        "p99_abs_error": float(np.quantile(errors, 0.99)) if n >= p99_min_samples else None,
        "max_abs_error": float(np.max(errors)),
        "p95_evaluated": n >= p95_min_samples,
        "p99_evaluated": n >= p99_min_samples,
    }


def _not_worse(candidate: float, incumbent: float, tolerance: float, epsilon: float) -> bool:
    limit = incumbent * (1.0 + tolerance) + epsilon
    return candidate <= limit


def _metric_compare(candidate: dict[str, Any], incumbent: dict[str, Any],
                    limits: dict[str, float], epsilon: float) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for metric, tolerance in limits.items():
        cand, inc = candidate.get(metric), incumbent.get(metric)
        if cand is None or inc is None:
            checks[metric] = {"evaluated": False, "passed": None,
                              "reason": "insufficient_samples"}
        else:
            passed = _not_worse(float(cand), float(inc), tolerance, epsilon)
            checks[metric] = {"evaluated": True, "passed": bool(passed),
                              "incumbent": float(inc), "candidate": float(cand),
                              "allowed_degradation": tolerance}
    return checks


def evaluate_model_candidate(y_true: Any, incumbent_pred: Any, candidate_pred: Any,
                              metadata: list[dict[str, Any]],
                              config: ModelGateConfig | None = None) -> dict[str, Any]:
    """Compare a candidate against the incumbent on identical, frozen samples.

    Status is PASS only when all required evidence is present and all checked
    constraints pass. Insufficient data returns REVIEW_REQUIRED, never PASS.
    Group metrics are reported separately for equipment, operating condition,
    and fault type; critical fault types receive a lower minimum sample floor.
    """
    cfg = config or ModelGateConfig()
    cfg.validate()
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    incumbent = np.asarray(incumbent_pred, dtype=float).reshape(-1)
    candidate = np.asarray(candidate_pred, dtype=float).reshape(-1)
    require(truth.size == incumbent.size == candidate.size == len(metadata) and truth.size > 0,
            "Predictions, truth, and metadata must have identical nonzero sample counts")
    require(np.isfinite(truth).all() and np.isfinite(incumbent).all() and np.isfinite(candidate).all(),
            "Gate inputs must be finite")

    base = regression_metrics(truth, incumbent, p95_min_samples=cfg.p95_min_samples,
                              p99_min_samples=cfg.p99_min_samples)
    cand = regression_metrics(truth, candidate, p95_min_samples=cfg.p95_min_samples,
                              p99_min_samples=cfg.p99_min_samples)
    global_limits = {
        "mae": cfg.mae_max_degradation,
        "p95_abs_error": cfg.p95_max_degradation,
        "p99_abs_error": cfg.p99_max_degradation,
        "max_abs_error": cfg.max_error_max_degradation,
    }
    global_checks = _metric_compare(cand, base, global_limits, cfg.zero_baseline_epsilon)
    failures: list[str] = []
    insufficient: list[str] = []
    for metric, check in global_checks.items():
        if check["evaluated"] and not check["passed"]:
            failures.append(f"global.{metric}")
        elif not check["evaluated"]:
            insufficient.append(f"global.{metric}")

    if cfg.absolute_max_error_limit is not None and cand["max_abs_error"] > cfg.absolute_max_error_limit:
        failures.append("global.absolute_max_error_limit")

    strata: dict[str, dict[str, Any]] = {}
    critical_values = set(cfg.critical_fault_types)
    for dimension in ("equipment_id", "operating_condition", "fault_type"):
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(metadata):
            value = str(row.get(dimension, "unknown"))
            grouped[value].append(index)
        strata[dimension] = {}
        for value, indices in sorted(grouped.items()):
            critical = dimension == "fault_type" and value in critical_values
            minimum = cfg.min_critical_stratum_samples if critical else cfg.min_stratum_samples
            if len(indices) < minimum:
                state = "REVIEW_REQUIRED" if critical else "INSUFFICIENT_NONCRITICAL"
                strata[dimension][value] = {"n": len(indices), "critical": critical,
                                            "status": state, "checks": {}}
                if critical:
                    insufficient.append(f"{dimension}={value}:n={len(indices)}")
                continue
            idx = np.asarray(indices, dtype=int)
            group_base = regression_metrics(truth[idx], incumbent[idx],
                                             p95_min_samples=cfg.p95_min_samples,
                                             p99_min_samples=cfg.p99_min_samples)
            group_cand = regression_metrics(truth[idx], candidate[idx],
                                            p95_min_samples=cfg.p95_min_samples,
                                            p99_min_samples=cfg.p99_min_samples)
            checks = _metric_compare(group_cand, group_base, {
                "mae": cfg.stratum_mae_max_degradation,
                "p95_abs_error": cfg.stratum_tail_max_degradation,
                "max_abs_error": cfg.max_error_max_degradation,
            }, cfg.zero_baseline_epsilon)
            group_failed = False
            group_insufficient = False
            for metric, check in checks.items():
                if check["evaluated"] and not check["passed"]:
                    failures.append(f"{dimension}={value}.{metric}")
                    group_failed = True
                elif not check["evaluated"]:
                    group_insufficient = True
            strata[dimension][value] = {
                "n": len(indices), "critical": critical,
                "status": "REJECT" if group_failed else ("REVIEW_REQUIRED" if group_insufficient else "PASS"),
                "incumbent": group_base, "candidate": group_cand, "checks": checks,
            }

    status = "REJECT" if failures else ("REVIEW_REQUIRED" if insufficient else "PASS")
    return {
        "status": status,
        "allow_upgrade": status == "PASS",
        "reason": ("; ".join(failures) if failures else
                   ("Insufficient evidence: " + "; ".join(insufficient) if insufficient else
                    "All configured regression and stratified checks passed")),
        "config": asdict(cfg),
        "global": {"incumbent": base, "candidate": cand, "checks": global_checks},
        "strata": strata,
        "failures": failures,
        "insufficient_evidence": insufficient,
        "evaluation_note": "Candidate and incumbent must be evaluated on the same frozen, leakage-free samples.",
    }
