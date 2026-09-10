from pathlib import Path
from agent.predictor_tool import PredictorTool
from agent.risk_tool import analyze as analyze_risk
from agent.anomaly_tool import analyze as analyze_anomaly
from agent.diagnosis_tool import diagnose


class OilTemperatureAgent:
    """End-to-end industrial oil-temperature forecasting and explainable diagnosis agent."""

    def __init__(self, checkpoint="agent/models/best_model.pt",
                 scaler_path="agent/models/scaler.pkl",
                 config_path="agent/models/config.yaml", device="cuda:0"):
        self.predictor = PredictorTool(checkpoint, scaler_path, config_path, device=device)
        self.risk_cfg = self.predictor.cfg["risk"]
        self.anomaly_cfg = self.predictor.cfg.get("anomaly", {})

    def run(self, history):
        pred = self.predictor.run(history)
        risk = analyze_risk(
            pred["forecast"],
            medium=float(self.risk_cfg["medium"]),
            high=float(self.risk_cfg["high"]),
        )
        anomaly = analyze_anomaly(
            history,
            target_index=self.predictor.target_idx,
            thresholds=self.anomaly_cfg.get("thresholds"),
        )
        diagnosis = diagnose(anomaly, risk)
        return {**pred, "risk": risk, "anomaly": anomaly, "diagnosis": diagnosis}

    def predict(self, history):
        result = self.run(history)
        return {
            "model": result["model"],
            "forecast_hours": result["forecast_hours"],
            "forecast": result["forecast"],
            "risk_level": result["risk"]["risk_level"],
            "peak_temperature": result["risk"]["max"],
            "peak_hour_ahead": result["risk"]["peak_hour_ahead"],
            "mean_temperature": result["risk"]["mean"],
            "max_rise_rate": result["risk"]["max_slope"],
            "rise": result["risk"]["rise"],
            "diagnosis_level": result["diagnosis"]["level"],
        }
