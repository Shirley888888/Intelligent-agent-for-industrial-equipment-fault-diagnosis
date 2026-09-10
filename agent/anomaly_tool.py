"""Rule-based historical thermal anomaly detection for the Agent.

The thresholds are configurable heuristics for this project, not protection-system
or industry-standard trip limits.
"""
import numpy as np

DEFAULT_THRESHOLDS = {
    "jump_c_per_hour": 5.0,
    "rise_6h_c": 3.0,
    "rise_12h_c": 5.0,
    "rise_24h_c": 8.0,
    "slope_6h_c_per_hour": 0.5,
    "slope_24h_c_per_hour": 0.35,
}


def _trend_slope(x):
    if len(x) < 2:
        return 0.0
    t = np.arange(len(x), dtype=float)
    return float(np.polyfit(t, x, 1)[0])


def analyze(history, target_index=-1, thresholds=None):
    arr = np.asarray(history, dtype=float)
    if arr.ndim == 1:
        x = arr
    elif arr.ndim == 2:
        if not (0 <= target_index < arr.shape[1] or target_index == -1):
            raise ValueError("target_index is out of range")
        x = arr[:, target_index]
    else:
        raise ValueError("history must be a 1-D OT series or a 2-D feature array")

    if len(x) < 24:
        raise ValueError("history must contain at least 24 observations")
    if not np.isfinite(x).all():
        raise ValueError("history contains NaN or infinite values")

    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update({k: float(v) for k, v in thresholds.items() if k in th})

    latest = float(x[-1])
    r6 = float(x[-1] - x[-7])
    r12 = float(x[-1] - x[-13])
    r24 = float(x[-1] - x[-25]) if len(x) >= 25 else float(x[-1] - x[0])
    s6 = _trend_slope(x[-6:])
    s24 = _trend_slope(x[-24:])
    diffs = np.diff(x)
    max_abs_jump = float(np.max(np.abs(diffs))) if len(diffs) else 0.0
    max_positive_slope = float(np.max(diffs)) if len(diffs) else 0.0

    reasons = []
    severe = False
    warning = False
    if max_abs_jump >= th["jump_c_per_hour"]:
        severe = True
        reasons.append(f"Abrupt temperature jump >= {th['jump_c_per_hour']:.1f} °C/h was detected.")
    if r24 >= th["rise_24h_c"]:
        severe = True
        reasons.append(f"24-hour temperature rise >= {th['rise_24h_c']:.1f} °C was detected.")
    if r12 >= th["rise_12h_c"]:
        warning = True
        reasons.append(f"12-hour temperature rise >= {th['rise_12h_c']:.1f} °C was detected.")
    if r6 >= th["rise_6h_c"]:
        warning = True
        reasons.append(f"6-hour temperature rise >= {th['rise_6h_c']:.1f} °C was detected.")
    if s6 >= th["slope_6h_c_per_hour"]:
        warning = True
        reasons.append(f"Recent 6-hour trend >= {th['slope_6h_c_per_hour']:.2f} °C/h was detected.")
    if s24 >= th["slope_24h_c_per_hour"]:
        warning = True
        reasons.append(f"Recent 24-hour trend >= {th['slope_24h_c_per_hour']:.2f} °C/h was detected.")

    if severe:
        status = "ANOMALY"
    elif warning:
        status = "WARNING"
    else:
        status = "NORMAL"
        reasons.append("No configured thermal anomaly rule was triggered.")

    # Transparent score: number of triggered rules, capped at 1.0.
    triggered = len(reasons) - (1 if status == "NORMAL" else 0)
    score = float(min(1.0, triggered / 4.0))

    return {
        "status": status,
        "score": score,
        "latest_temperature": latest,
        "rise_6h": r6,
        "rise_12h": r12,
        "rise_24h": r24,
        "slope_6h": s6,
        "slope_24h": s24,
        "max_abs_jump": max_abs_jump,
        "max_positive_slope": max_positive_slope,
        "reasons": reasons,
        "thresholds": th,
    }
