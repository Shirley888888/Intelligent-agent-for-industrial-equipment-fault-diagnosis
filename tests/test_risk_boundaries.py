"""Boundary tests for the dynamic quantile risk engine."""  # modify:dynamic-risk

import os
import sys
import yaml  # modify:config-only-hyperparams

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent.risk_tool import analyze, calculate_dynamic_thresholds  # modify:dynamic-risk


def main():
    # modify:config-only-hyperparams - tests use the same configured quantiles as the Agent.
    with open(os.path.join(ROOT, "agent", "models", "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rcfg = cfg["risk"]
    threshold_args = dict(data_path=os.path.join(ROOT, rcfg["data_path"]), target=cfg["target"],
                          train_ratio=rcfg["train_ratio"], forecast_horizon=rcfg["forecast_horizon"],
                          quantiles=rcfg["quantiles"])
    thresholds = calculate_dynamic_thresholds(**threshold_args)  # modify:dynamic-risk
    t = thresholds["temperature"]  # modify:dynamic-risk

    # modify:dynamic-risk - verify calculated boundaries instead of legacy fixed 75/90 C.
    cases = [
        ("below_medium", [t["medium"] - 0.01] * 24, "LOW"),
        ("medium_boundary", [t["medium"]] * 24, "MEDIUM"),
        ("below_high", [t["high"] - 0.01] * 24, "MEDIUM"),
        ("high_boundary", [t["high"]] * 24, "HIGH"),
    ]

    print("=== Dynamic risk boundary tests ===")  # modify:dynamic-risk
    print({"temperature_thresholds": t})  # modify:dynamic-risk
    passed = 0

    for name, forecast, expected in cases:
        result = analyze(forecast, thresholds=thresholds)  # modify:dynamic-risk
        actual = result["risk_level"]
        ok = actual == expected
        print({
            "case": name,
            "max": max(forecast),
            "expected": expected,
            "actual": actual,
            "status": "PASS" if ok else "FAIL",
        })
        if not ok:
            raise AssertionError(
                f"{name}: expected {expected}, got {actual}"
            )
        passed += 1

    print(f"ALL RISK BOUNDARY TESTS PASSED ({passed}/{len(cases)})")
    # modify:regression-tests - training boundaries, degenerate statistics and provenance.
    import numpy as np  # modify:regression-tests
    import pandas as pd  # modify:regression-tests
    from unittest.mock import patch  # modify:regression-tests
    from agent.risk_tool import risk_level  # modify:regression-tests
    for metric, key in (("rise", "rise"), ("slope", "max_slope")):  # modify:regression-tests
        for level in ("medium", "high"):  # modify:regression-tests
            assert risk_level(-100, thresholds=thresholds, **{key: thresholds[metric][level]}) == level.upper()  # modify:regression-tests
    with patch("agent.risk_tool.pd.read_csv", return_value=pd.DataFrame({"OT": np.arange(30.)})):  # modify:regression-tests
        short = calculate_dynamic_thresholds(**threshold_args)  # modify:regression-tests
        assert short["source"]["train_rows"] == 21  # modify:regression-tests
        assert short["source"]["forecast_horizon"] == 21  # modify:regression-tests
    with patch("agent.risk_tool.pd.read_csv", return_value=pd.DataFrame({"OT": [20.] * 100})):  # modify:regression-tests
        constant = calculate_dynamic_thresholds(**threshold_args)  # modify:regression-tests
        assert analyze([20.] * 24, thresholds=constant)["risk_level"] == "LOW"  # modify:regression-tests
        assert analyze([21.] * 24, thresholds=constant)["risk_level"] == "HIGH"  # modify:regression-tests
    try:  # modify:config-only-hyperparams
        analyze([20.] * 24)
        raise AssertionError("analyze() must not fabricate thresholds without configured statistics")
    except ValueError:
        pass
    print("ADDITIONAL RISK REGRESSION TESTS PASSED")  # modify:regression-tests


if __name__ == "__main__":
    main()
