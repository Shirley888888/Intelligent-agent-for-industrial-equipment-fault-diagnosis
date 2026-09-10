"""Boundary tests for the configurable risk engine.

These tests do not alter or retrain the forecasting model. They verify that
the configured temperature thresholds map to LOW / MEDIUM / HIGH correctly.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent.risk_tool import analyze


def main():
    # Exact boundary cases plus values just below each boundary.
    cases = [
        ("below_medium", [74.9] * 24, "LOW"),
        ("medium_boundary", [75.0] * 24, "MEDIUM"),
        ("below_high", [89.9] * 24, "MEDIUM"),
        ("high_boundary", [90.0] * 24, "HIGH"),
    ]

    print("=== Risk boundary tests ===")
    passed = 0

    for name, forecast, expected in cases:
        result = analyze(forecast)
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


if __name__ == "__main__":
    main()
