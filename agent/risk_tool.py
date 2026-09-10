import numpy as np

def risk_level(peak_temperature, medium=75.0, high=90.0):
    if peak_temperature >= high:
        return "HIGH"
    if peak_temperature >= medium:
        return "MEDIUM"
    return "LOW"

def analyze(forecast, medium=75.0, high=90.0):
    x = np.asarray(forecast, dtype=float)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("forecast must be a non-empty 1-D sequence")
    if not np.isfinite(x).all():
        raise ValueError("forecast contains NaN or infinite values")

    peak = float(np.max(x))
    peak_hour = int(np.argmax(x)) + 1
    mean = float(np.mean(x))
    rise = float(peak - x[0])
    max_slope = float(np.max(np.diff(x))) if len(x) > 1 else 0.0

    return {
        "mean": mean,
        "max": peak,
        "rise": rise,
        "max_slope": max_slope,
        "risk_level": risk_level(peak, medium, high),
        "peak_hour_ahead": peak_hour
    }

# Backward-compatible names
def summarize(forecast, medium=75.0, high=90.0):
    r = analyze(forecast, medium, high)
    return {
        "risk_level": r["risk_level"],
        "peak_temperature": r["max"],
        "peak_hour_ahead": r["peak_hour_ahead"],
        "mean_temperature": r["mean"],
        "max_rise_rate": r["max_slope"],
        "rise": r["rise"]
    }
