"""Proactive training trigger: performance drift + distribution shift + schedule."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Sequence
import numpy as np

@dataclass
class TriggerConfig:
    min_feedback: int = 30
    rolling_window: int = 50
    performance_ratio: float = 1.20
    consecutive_checks: int = 2
    psi_threshold: float = 0.20
    scheduled_days: int = 14
    cooldown_hours: int = 24

@dataclass
class TriggerDecision:
    should_train: bool
    reasons: list[str]
    recent_mae: float | None
    baseline_mae: float | None
    psi_max: float | None
    next_check: str
    evidence: dict

def population_stability_index(reference, current, bins=10):
    """PSI for one feature; quantile bins are fixed from reference data."""
    ref=np.asarray(reference,dtype=float).ravel()
    cur=np.asarray(current,dtype=float).ravel()
    ref=ref[np.isfinite(ref)]; cur=cur[np.isfinite(cur)]
    if len(ref)<bins*5 or len(cur)<bins*2: return None
    edges=np.unique(np.quantile(ref,np.linspace(0,1,bins+1)))
    if len(edges)<3: return 0.0
    edges[0],edges[-1]=-np.inf,np.inf
    a=np.histogram(ref,edges)[0]/len(ref)
    b=np.histogram(cur,edges)[0]/len(cur)
    eps=1e-6
    return float(np.sum((b-a)*np.log((b+eps)/(a+eps))))

def evaluate_training_trigger(*, errors: Sequence[float], reference_features=None,
                               recent_features=None, last_train_time=None,
                               now=None, consecutive_bad_checks=0,
                               config: TriggerConfig | None=None) -> TriggerDecision:
    cfg=config or TriggerConfig()
    now=now or datetime.now(timezone.utc)
    err=np.asarray(errors,dtype=float)
    err=err[np.isfinite(err)]
    reasons=[]; recent=baseline=psi_max=None
    if len(err)>=cfg.min_feedback:
        w=min(cfg.rolling_window,len(err))
        recent=float(np.mean(err[-w:]))
        # Baseline uses earlier matured feedback, excluding the recent window.
        older=err[:-w]
        if len(older)>=max(10,cfg.min_feedback//2):
            baseline=float(np.mean(older))
            if recent > baseline*cfg.performance_ratio:
                reasons.append("performance_drift")
                if consecutive_bad_checks+1 < cfg.consecutive_checks:
                    reasons[-1]="performance_drift_pending_confirmation"
    psi_values=[]
    if reference_features is not None and recent_features is not None:
        r=np.asarray(reference_features); c=np.asarray(recent_features)
        if r.ndim==1: r=r[:,None]
        if c.ndim==1: c=c[:,None]
        if r.ndim==2 and c.ndim==2 and r.shape[1]==c.shape[1]:
            for j in range(r.shape[1]):
                value=population_stability_index(r[:,j],c[:,j])
                if value is not None: psi_values.append(value)
        if psi_values:
            psi_max=max(psi_values)
            if psi_max>=cfg.psi_threshold: reasons.append("data_distribution_shift")
    if last_train_time is not None:
        elapsed=(now-last_train_time).total_seconds()/86400
        if elapsed>=cfg.scheduled_days: reasons.append("scheduled_retraining")
    should=any(x!="performance_drift_pending_confirmation" for x in reasons)
    if last_train_time is not None and (now-last_train_time).total_seconds()<cfg.cooldown_hours*3600:
        should=False; reasons.append("cooldown_active")
    return TriggerDecision(should,reasons,recent,baseline,psi_max,now.isoformat(),
        {"n_feedback":int(len(err)),"config":asdict(cfg),
         "consecutive_bad_checks":int(consecutive_bad_checks),
         "psi_by_feature":psi_values})
