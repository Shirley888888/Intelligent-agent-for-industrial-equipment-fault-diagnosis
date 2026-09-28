"""Explainable diagnosis decision layer combining historical anomaly and forecast risk."""


def diagnose(anomaly, risk):
    a = anomaly["status"]
    r = risk["risk_level"]
    evidence = []

    if anomaly.get("reasons"):
        evidence.extend(anomaly["reasons"])
    evidence.append(f"Forecast peak is {risk['max']:.2f} °C.")
    evidence.append(f"Forecast maximum positive slope is {risk['max_slope']:.2f} °C/h.")

    # Conservative precedence: explicit historical anomaly first, then warning,
    # then forecast thermal risk. This is a decision aid, not a protection rule.
    if a == "ANOMALY":
        level = "ANOMALY"
        diagnosis = "Abnormal historical thermal behavior detected."
        recommendation = "Review recent operating conditions, load/cooling behavior, and sensor quality; escalate for engineering inspection if the pattern persists."
    elif r == "HIGH":
        level = "ANOMALY"
        diagnosis = "Forecast indicates high thermal risk within the next 24 hours."
        recommendation = "Increase monitoring and review operating/load and cooling conditions; follow the site's approved operating and inspection procedures."
    elif a == "WARNING" or r == "MEDIUM":
        level = "WARNING"
        diagnosis = "A developing thermal risk pattern was detected."
        recommendation = "Increase monitoring frequency and review recent load, cooling, ambient conditions, and sensor behavior."
    else:
        level = "NORMAL"
        diagnosis = "Temperature behavior is currently stable with no configured historical anomaly or forecast thermal risk trigger."
        recommendation = "Continue routine monitoring and reassess with the next 96-hour observation window."

    return {
        "level": level,
        "diagnosis": diagnosis,
        "reason": diagnosis,
        "evidence": evidence,
        "recommendation": recommendation,
    }
