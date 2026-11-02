from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import math

from .common import require


@dataclass
class RuntimeGuardConfig:
    ood_warn: float = 0.70
    ood_critical: float = 0.90
    drift_warn: float = 0.20
    drift_critical: float = 0.30
    min_calibrated_confidence: float = 0.70
    calibration_id: str = "UNSET"

    def validate(self) -> None:
        for name in ("ood_warn", "ood_critical", "drift_warn", "drift_critical",
                     "min_calibrated_confidence"):
            value = getattr(self, name)
            require(math.isfinite(value) and 0 <= value <= 1, f"{name} must be in [0,1]")
        require(self.ood_warn < self.ood_critical, "ood_warn must be lower than ood_critical")
        require(self.drift_warn < self.drift_critical, "drift_warn must be lower than drift_critical")


def runtime_safety_decision(*, ood_score: float | None, drift_score: float | None,
                            confidence: float | None, confidence_calibrated: bool,
                            input_quality: str = "valid", sensor_conflict: bool = False,
                            config: RuntimeGuardConfig | None = None) -> dict[str, Any]:
    """Deterministic guard. Scores must be supplied by independently validated detectors.

    This function does not calculate OOD, drift, or calibrated confidence.
    """
    cfg = config or RuntimeGuardConfig()
    cfg.validate()
    reasons: list[str] = []
    hard_stop = False
    conservative = False

    if input_quality != "valid":
        reasons.append(f"input_quality={input_quality}")
        hard_stop = True
    if sensor_conflict:
        reasons.append("sensor_conflict")
        hard_stop = True
    if ood_score is None or drift_score is None or confidence is None:
        reasons.append("missing_runtime_signal")
        hard_stop = True
    else:
        for name, value in (("ood_score", ood_score), ("drift_score", drift_score),
                            ("confidence", confidence)):
            require(math.isfinite(value) and 0 <= value <= 1, f"{name} must be finite and in [0,1]")
        if ood_score >= cfg.ood_critical:
            reasons.append("critical_ood")
            hard_stop = True
        elif ood_score >= cfg.ood_warn:
            reasons.append("elevated_ood")
            conservative = True
        if drift_score >= cfg.drift_critical:
            reasons.append("critical_distribution_drift")
            hard_stop = True
        elif drift_score >= cfg.drift_warn:
            reasons.append("distribution_drift")
            conservative = True
        if confidence < cfg.min_calibrated_confidence:
            reasons.append("low_confidence")
            conservative = True
    if not confidence_calibrated:
        reasons.append("confidence_not_calibrated")
        conservative = True
        hard_stop = True
    if cfg.calibration_id == "UNSET":
        reasons.append("calibration_not_configured")
        hard_stop = True

    if hard_stop:
        mode = "human_review"
        allow_definitive_diagnosis = False
        allow_automatic_control = False
    elif conservative:
        mode = "conservative"
        allow_definitive_diagnosis = False
        allow_automatic_control = False
    else:
        mode = "normal"
        allow_definitive_diagnosis = True
        allow_automatic_control = True

    return {
        "mode": mode,
        "allow_definitive_diagnosis": allow_definitive_diagnosis,
        "allow_automatic_control": allow_automatic_control,
        "request_human_review": mode != "normal",
        "reasons": reasons or ["signals_within_configured_limits"],
        "config": asdict(cfg),
        "decision_source": "deterministic_runtime_guard",
        "note": "OOD/drift/confidence scores are external signals; calibrate thresholds for each task and equipment family.",
    }
