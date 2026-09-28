import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.anomaly_tool import analyze
from agent.diagnosis_tool import diagnose


def risk(level, peak):
    return {"risk_level": level, "max": peak, "max_slope": 0.2}

# Historical patterns: stable, warning rise, severe jump.
stable = np.ones(96) * 20.0
warning = np.ones(96) * 20.0
warning[-6:] = np.linspace(20.0, 23.2, 6)
severe = np.ones(96) * 20.0
severe[-1] = 27.0

cases = [
    ("stable", stable, "NORMAL", "NORMAL"),
    ("warning", warning, "WARNING", "WARNING"),
    ("severe", severe, "ANOMALY", "ANOMALY"),
]
for name, x, expected_a, expected_d in cases:
    a = analyze(x)
    d = diagnose(a, risk("LOW", 30.0))
    assert a["status"] == expected_a, (name, a)
    assert d["level"] == expected_d, (name, d)
    print({"case": name, "anomaly": a["status"], "diagnosis": d["level"], "status": "PASS"})

# Forecast risk can independently elevate diagnosis.
a = analyze(stable)
d = diagnose(a, risk("HIGH", 95.0))
assert d["level"] == "ANOMALY"
print({"case": "forecast_high", "diagnosis": d["level"], "status": "PASS"})
print("ALL DIAGNOSIS TESTS PASSED (4/4)")
