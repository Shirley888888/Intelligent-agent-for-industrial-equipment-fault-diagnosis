from agent.predictor_tool import PredictorTool
from agent.risk_tool import analyze as analyze_risk
from agent.risk_tool import calculate_dynamic_thresholds
from agent.anomaly_tool import analyze as analyze_anomaly
from agent.diagnosis_tool import diagnose
from agent.rag_tool import RAGTool  # modify:rag-observation


class OilTemperatureAgent:
    """End-to-end Forecast -> Risk -> RAG -> multi-step ReAct industrial Agent."""

    def __init__(self, checkpoint=None, scaler_path=None,
                 config_path="agent/models/config.yaml", device="cuda:0"):
        self.predictor = PredictorTool(checkpoint, scaler_path, config_path, device=device)
        self.risk_cfg = self.predictor.cfg["risk"]
        self.anomaly_cfg = self.predictor.cfg.get("anomaly", {})
        self.rag = RAGTool(self.predictor.cfg["rag"])  # modify:rag-observation
        self.react_max_steps = int(self.predictor.cfg["react"]["max_steps"])  # modify:react-audit
        self.risk_thresholds = calculate_dynamic_thresholds(
            data_path=self.risk_cfg["data_path"], target=self.predictor.target,
            train_ratio=float(self.risk_cfg["train_ratio"]),
            forecast_horizon=int(self.risk_cfg["forecast_horizon"]),
            quantiles=self.risk_cfg["quantiles"],
            print_stats=True,  # modify:dynamic-risk - startup audit log
        )

    def run(self, history):
        observations = []  # modify:react-audit - full tool Observation log.

        pred = self.predictor.run(history)
        observations.append({"step": 1, "thought": "Forecast the next oil-temperature horizon.",
                             "action": "Forecast", "observation": pred})

        risk = analyze_risk(pred["forecast"], thresholds=self.risk_thresholds)
        observations.append({"step": 2, "thought": "Evaluate forecast thermal indicators against training quantiles.",
                             "action": "Risk", "observation": risk})

        anomaly = analyze_anomaly(history, target_index=self.predictor.target_idx,
                                  thresholds=self.anomaly_cfg.get("thresholds"))
        query = (f"transformer oil temperature risk {risk['risk_level']} peak {risk['max']:.2f} "
                 f"rise {risk['rise']:.2f} slope {risk['max_slope']:.2f}; historical status {anomaly['status']}")
        evidence = self.rag.retrieve(query)
        observations.append({"step": 3, "thought": "Retrieve maintenance evidence relevant to the forecast and risk.",
                             "action": "RAG", "observation": evidence})

        # modify:react-audit - second retrieval is conditional but keeps the same RAG core/tool.
        if self.react_max_steps >= 4 and not any(x["accepted"] for x in evidence):
            second = self.rag.retrieve(f"oil temperature {risk['risk_level']} cooling load monitoring")
            evidence.extend(second)
            observations.append({"step": 4, "thought": "No first-pass evidence passed the similarity filter; refine retrieval.",
                                 "action": "RAG", "observation": second})

        diagnosis = diagnose(anomaly, risk)
        accepted = [x for x in evidence if x["accepted"]]
        evidence_text = "; ".join(f"{x['id']}: {x['text']}" for x in accepted) or "No RAG evidence passed the configured similarity threshold."
        final_text = (f"Risk={risk['risk_level']}. {diagnosis['diagnosis']} "
                      f"Forecast peak={risk['max']:.2f}°C, rise={risk['rise']:.2f}°C, "
                      f"max positive slope={risk['max_slope']:.2f}°C/h. "
                      f"Evidence: {evidence_text} Recommendation: {diagnosis['recommendation']}")

        return {**pred, "risk": risk, "anomaly": anomaly, "rag_evidence": evidence,
                "react_observations": observations, "diagnosis": diagnosis,
                "agent_final_output": final_text}  # modify:complete-output

    def predict(self, history):
        result = self.run(history)
        return {
            "model": result["model"], "forecast_hours": result["forecast_hours"],
            "forecast": result["forecast"], "risk_level": result["risk"]["risk_level"],
            "peak_temperature": result["risk"]["max"], "peak_hour_ahead": result["risk"]["peak_hour_ahead"],
            "mean_temperature": result["risk"]["mean"], "max_rise_rate": result["risk"]["max_slope"],
            "rise": result["risk"]["rise"], "diagnosis_level": result["diagnosis"]["level"],
        }
