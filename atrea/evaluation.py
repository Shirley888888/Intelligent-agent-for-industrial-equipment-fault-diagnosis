"""Paired evaluation on identical frozen windows; bootstrap CI + optional Wilcoxon."""
from __future__ import annotations
import numpy as np
def regression_metrics(y_true,y_pred):
    y=np.asarray(y_true,float).reshape(-1); p=np.asarray(y_pred,float).reshape(-1)
    if y.size!=p.size or not y.size: raise ValueError("truth/prediction shape mismatch")
    e=p-y
    return {"mae":float(np.mean(np.abs(e))),"rmse":float(np.sqrt(np.mean(e**2))),
            "mape":float(np.mean(np.abs(e)/(np.abs(y)+1e-8))*100)}
def paired_bootstrap(candidate_errors, incumbent_errors, n_boot=5000, seed=42):
    a=np.asarray(candidate_errors,float).reshape(-1); b=np.asarray(incumbent_errors,float).reshape(-1)
    if a.size!=b.size or a.size<20: raise ValueError("Need >=20 paired test errors")
    d=a-b; rng=np.random.default_rng(seed)
    samples=np.mean(d[rng.integers(0,len(d),(n_boot,len(d)))],axis=1)
    lo,hi=np.quantile(samples,[.025,.975])
    try:
        from scipy.stats import wilcoxon
        stat,p=wilcoxon(a,b,alternative="less",zero_method="wilcox")
        test={"name":"wilcoxon_one_sided","p_value":float(p)}
    except Exception:
        test={"name":"paired_bootstrap_CI_only","p_value":None}
    return {"mean_mae_difference_candidate_minus_incumbent":float(np.mean(d)),
            "ci95":[float(lo),float(hi)],"significant_improvement":bool(hi<0 and (test["p_value"] is None or test["p_value"]<.05)),
            "test":test,"n":int(len(d))}
def compare_candidate(y_true,incumbent_pred,candidate_pred):
    y=np.asarray(y_true,float).reshape(-1); a=np.abs(y-np.asarray(incumbent_pred,float).reshape(-1))
    b=np.abs(y-np.asarray(candidate_pred,float).reshape(-1))
    if not (len(y)==len(a)==len(b)): raise ValueError("Predictions must share frozen test indices")
    return {"incumbent":regression_metrics(y,incumbent_pred),
            "candidate":regression_metrics(y,candidate_pred),
            "paired_test":paired_bootstrap(b,a)}
