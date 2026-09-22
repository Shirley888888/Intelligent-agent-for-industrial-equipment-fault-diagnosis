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

    # modify:dynamic-risk - simple fallback uses the bundled ETTh1 path instead of failing immediately.
    if not path.exists():
        path = Path(__file__).resolve().parents[1] / "data" / "ETTh1.csv"  # modify:dynamic-risk

    df = pd.read_csv(path)  # modify:dynamic-risk
    if target not in df.columns:  # modify:dynamic-risk
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # modify:dynamic-risk
        target = numeric_cols[-1] if numeric_cols else df.columns[-1]  # modify:dynamic-risk

    if not 0 < float(train_ratio) <= 1:  # modify:training-boundary
        raise ValueError("train_ratio must be in (0, 1].")  # modify:training-boundary
    n_train = int(len(df) * float(train_ratio))  # modify:training-boundary - never extend into validation.
    train_ot = pd.to_numeric(df.iloc[:n_train][target], errors="coerce").dropna().to_numpy(float)  # modify:dynamic-risk
    train_ot = train_ot[np.isfinite(train_ot)]  # modify:finite-statistics
    if len(train_ot) == 0:  # modify:training-boundary
        raise ValueError("No finite training OT values; cannot fabricate risk thresholds.")  # modify:training-boundary
    horizon = min(max(1, int(forecast_horizon)), len(train_ot))  # modify:short-training

    peaks, rises, slopes = [], [], []  # modify:dynamic-risk
    for start in range(0, len(train_ot) - horizon + 1):  # modify:dynamic-risk
        window = train_ot[start:start + horizon]  # modify:dynamic-risk
        peaks.append(float(np.max(window)))  # modify:dynamic-risk
        rises.append(float(np.max(window) - window[0]))  # modify:dynamic-risk
        slopes.append(max(0.0, float(np.max(np.diff(window)))) if len(window) > 1 else 0.0)  # modify:positive-slope

    # modify:dynamic-risk - retain a usable path for very short custom datasets.
    if not peaks:
        peaks = train_ot.tolist()  # modify:dynamic-risk
        rises = [0.0]  # modify:dynamic-risk
        slopes = [0.0]  # modify:dynamic-risk

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
