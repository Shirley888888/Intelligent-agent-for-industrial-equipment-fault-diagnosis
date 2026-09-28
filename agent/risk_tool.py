from pathlib import Path  # modify:dynamic-risk
import numpy as np
import pandas as pd  # modify:dynamic-risk


def calculate_dynamic_thresholds(data_path, target, train_ratio, forecast_horizon, quantiles, print_stats=False):  # modify:config-only-hyperparams
    """Calculate risk thresholds from chronological training data only."""  # modify:dynamic-risk
    if quantiles is None:  # modify:config-only-hyperparams
        raise ValueError("risk.quantiles must be provided by config; risk thresholds are not hardcoded in code.")  # modify:config-only-hyperparams

    path = Path(data_path)  # modify:dynamic-risk
    if not path.exists():  # modify:dynamic-risk
        project_path = Path(__file__).resolve().parents[1] / path  # modify:dynamic-risk
        path = project_path if project_path.exists() else path  # modify:dynamic-risk

    # modify:strict-calibration - fail instead of silently changing data/target.
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column missing: {target}")

    if not 0 < float(train_ratio) <= 1:  # modify:training-boundary
        raise ValueError("train_ratio must be in (0, 1].")  # modify:training-boundary
    n_train = int(len(df) * float(train_ratio))  # modify:training-boundary - never extend into validation.
    train_ot = pd.to_numeric(df.iloc[:n_train][target], errors="raise").to_numpy(float)  # modify:dynamic-risk
    if not np.isfinite(train_ot).all():
        raise ValueError("Nonfinite training values; do not compress time.")
    # train_ot remains chronological  # modify:finite-statistics
    if len(train_ot) == 0:  # modify:training-boundary
        raise ValueError("No finite training OT values; cannot fabricate risk thresholds.")  # modify:training-boundary
    horizon = min(max(1, int(forecast_horizon)), len(train_ot))  # modify:short-training

    # modify:audit-vectorize - same 24h windows and quantiles, vectorized without case input.
    windows = np.lib.stride_tricks.sliding_window_view(train_ot, horizon)
    peaks = windows.max(axis=1)
    rises = peaks - windows[:, 0]
    slopes = np.maximum(np.diff(windows, axis=1).max(axis=1), 0.0) if horizon > 1 else np.zeros(len(windows))

    metric_values = {"temperature": peaks, "rise": rises, "slope": slopes}  # modify:dynamic-risk
    thresholds = {}  # modify:dynamic-risk
    for metric, values in metric_values.items():  # modify:dynamic-risk
        qcfg = quantiles.get(metric)  # modify:config-only-hyperparams
        if not isinstance(qcfg, dict) or "medium" not in qcfg or "high" not in qcfg:  # modify:config-only-hyperparams
            raise ValueError(f"risk.quantiles.{metric}.medium/high must be defined in config.")  # modify:config-only-hyperparams
        q_medium = float(qcfg["medium"])  # modify:config-only-hyperparams
        q_high = float(qcfg["high"])  # modify:config-only-hyperparams
        if not 0 <= q_medium < q_high <= 1:  # modify:quantile-validation
            raise ValueError("Quantiles must satisfy 0 <= medium < high <= 1.")  # modify:quantile-validation
        arr = np.asarray(values, dtype=float)  # modify:dynamic-risk
        thresholds[metric] = {  # modify:dynamic-risk
            "medium": float(np.quantile(arr, q_medium)),  # modify:dynamic-risk
            "high": float(np.quantile(arr, q_high)),  # modify:dynamic-risk
            "medium_quantile": q_medium,  # modify:dynamic-risk
            "high_quantile": q_high,  # modify:dynamic-risk
            "constant": bool(np.ptp(arr) == 0),  # modify:constant-statistics
        }

    thresholds["source"] = {  # modify:dynamic-risk
        "data_path": str(path),  # modify:dynamic-risk
        "train_rows": int(len(train_ot)),  # modify:dynamic-risk
        "forecast_horizon": int(horizon),  # modify:dynamic-risk
    }
    if print_stats:  # modify:dynamic-risk - required startup statistics log.
        print("[RiskThresholds] training-set quantile statistics")
        for metric in ("temperature", "rise", "slope"):
            t = thresholds[metric]
            print(f"  {metric}: q{t['medium_quantile']:.2f}={t['medium']:.6f}, q{t['high_quantile']:.2f}={t['high']:.6f}")
    return thresholds  # modify:dynamic-risk


def risk_level(peak_temperature, medium=None, high=None, rise=0.0, max_slope=0.0, thresholds=None):
    """Map temperature/rise/slope indicators to LOW/MEDIUM/HIGH risk."""  # modify:dynamic-risk
    if thresholds is None:  # modify:dynamic-risk
        if medium is not None and high is not None:  # modify:dynamic-risk - retain legacy caller compatibility.
            thresholds = {  # modify:dynamic-risk
                "temperature": {"medium": float(medium), "high": float(high)},  # modify:dynamic-risk
                "rise": {"medium": float("inf"), "high": float("inf")},  # modify:dynamic-risk
                "slope": {"medium": float("inf"), "high": float("inf")},  # modify:dynamic-risk
            }
        else:
            raise ValueError("Dynamic risk thresholds must be calculated from config and passed to risk_level().")  # modify:config-only-hyperparams

    values = {  # modify:dynamic-risk
        "temperature": float(peak_temperature),  # modify:dynamic-risk
        "rise": float(rise),  # modify:dynamic-risk
        "slope": float(max_slope),  # modify:dynamic-risk
    }
    # modify:constant-statistics - equality to a constant training reference is not an anomaly.
    def crossed(name, level):  # modify:constant-statistics
        bound = float(thresholds[name][level])  # modify:constant-statistics
        return values[name] > bound if thresholds[name].get("constant", False) else values[name] >= bound  # modify:constant-statistics
    if any(crossed(name, "high") for name in values):  # modify:constant-statistics
        return "HIGH"
    if any(crossed(name, "medium") for name in values):  # modify:constant-statistics
        return "MEDIUM"
    return "LOW"


def analyze(forecast, medium=None, high=None, thresholds=None):
    if thresholds is None and (medium is None or high is None):  # modify:threshold-provenance
        raise ValueError("analyze() requires dynamic thresholds from the configured training-set quantiles.")  # modify:config-only-hyperparams
    x = np.asarray(forecast, dtype=float)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("forecast must be a non-empty 1-D sequence")
    if not np.isfinite(x).all():
        raise ValueError("forecast contains NaN or infinite values")

    peak = float(np.max(x))
    peak_hour = int(np.argmax(x)) + 1
    mean = float(np.mean(x))
    rise = float(peak - x[0])
    max_slope = max(0.0, float(np.max(np.diff(x)))) if len(x) > 1 else 0.0  # modify:positive-slope

    level = risk_level(  # modify:dynamic-risk
        peak, medium=medium, high=high, rise=rise, max_slope=max_slope, thresholds=thresholds  # modify:dynamic-risk
    )
    return {
        "mean": mean,
        "max": peak,
        "rise": rise,
        "max_slope": max_slope,
        "risk_level": level,  # modify:dynamic-risk
        "peak_hour_ahead": peak_hour,
        "thresholds": thresholds,  # modify:dynamic-risk
    }


# Backward-compatible names
def summarize(forecast, medium=None, high=None, thresholds=None):
    r = analyze(forecast, medium, high, thresholds=thresholds)  # modify:dynamic-risk
    return {
        "risk_level": r["risk_level"],
        "peak_temperature": r["max"],
        "peak_hour_ahead": r["peak_hour_ahead"],
        "mean_temperature": r["mean"],
        "max_rise_rate": r["max_slope"],
        "rise": r["rise"],
        "thresholds": r["thresholds"],  # modify:dynamic-risk
    }


# modify:freeze-risk - frozen artifact binds data hash and calibration configuration.
def load_frozen_thresholds(cfg):
    import json, hashlib
    root = Path(__file__).resolve().parents[1]
    rc = cfg["risk"]
    path = root / rc["frozen_path"]
    if not path.exists():
        raise FileNotFoundError("Run python calibrate_risk.py before running cases.")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    expected = {"target": cfg["target"], "train_ratio": rc["train_ratio"],
                "forecast_horizon": rc["forecast_horizon"], "quantiles": rc["quantiles"]}
    if artifact["calibration_config"] != expected:
        raise ValueError("Risk config differs from frozen calibration.")
    if hashlib.sha256((root / rc["data_path"]).read_bytes()).hexdigest() != artifact["dataset_sha256"]:
        raise ValueError("Dataset differs from frozen calibration.")
    # modify:audit-freeze-integrity - verify stored values as well as configuration/hash.
    computed = calculate_dynamic_thresholds(root / rc["data_path"], cfg["target"],
        rc["train_ratio"], rc["forecast_horizon"], rc["quantiles"])
    for metric in ("temperature", "rise", "slope"):
        if computed[metric] != artifact["thresholds"][metric]:
            raise ValueError(f"Frozen {metric} thresholds do not match training quantiles.")
    if computed["source"]["train_rows"] != artifact["thresholds"]["source"]["train_rows"]:
        raise ValueError("Frozen calibration row count mismatch.")
    return artifact["thresholds"]
