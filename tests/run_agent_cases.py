import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import pandas as pd
from agent.agent import OilTemperatureAgent

FEATURES=["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]


def main(agent=None):
    project_root=Path(__file__).resolve().parents[1]
    cases_root=Path(__file__).resolve().parent/"cases"
    output_root=project_root/"output"  # modify:case-persistence
    output_root.mkdir(parents=True, exist_ok=True)
    agent=agent or OilTemperatureAgent()
    case_order=["stable","slow_rise","fast_rise"]
    all_results={}

    for case_name in case_order:
        input_path=cases_root/case_name/"input.csv"
        df=pd.read_csv(input_path)
        history=df[FEATURES].tail(96).values
        result=agent.run(history)  # modify:full-react-case - complete Forecast/Risk/RAG/ReAct path.
        risk=result["risk"]
        payload={
            "case_name": case_name,
            "raw_input_signal": history.tolist(),
            "forecast_output": result["forecast"],
            "temperature_metric": risk["max"],
            "temperature_rise_metric": risk["rise"],
            "slope_metric": risk["max_slope"],
            "risk_intermediate_values": {"mean": risk["mean"], "peak_hour_ahead": risk["peak_hour_ahead"],
                                         "thresholds": risk["thresholds"]},
            "rag_evidence": result["rag_evidence"],
            "react_observations": result["react_observations"],
            "risk_level": risk["risk_level"],
            "agent_final_output": result["agent_final_output"],
            "model": result["model"],
            "anomaly": result["anomaly"],
            "diagnosis": result["diagnosis"],
        }
        out=output_root/f"case_{case_name}.json"
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        all_results[case_name]=payload
        print(f"[CaseResult] {case_name}: risk={risk['risk_level']}, peak={risk['max']:.3f}, "
              f"rise={risk['rise']:.3f}, slope={risk['max_slope']:.3f} -> {out.relative_to(project_root)}")

    (output_root/"all_case_results.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    expected={"stable":"LOW","slow_rise":"MEDIUM","fast_rise":"HIGH"}  # modify:case-verification - expected demo semantics, not thresholds.
    actual={k:v["risk_level"] for k,v in all_results.items()}
    if actual != expected:
        raise AssertionError(f"Case risk separation failed: expected {expected}, got {actual}. Adjust config quantiles only.")
    print(f"[CaseSummary] {actual}")
    return all_results

if __name__=="__main__": main()
