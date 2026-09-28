import argparse, json
from pathlib import Path
import pandas as pd

FEATURES = ["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]

def load_history(csv_path):
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")
    if "date" not in df.columns:
        raise ValueError("CSV must include date timestamps to verify hourly continuity.")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="raise")
        # modify:audit-time-grid - duplicates/gaps break hourly slope interpretation.
        if df["date"].isna().any() or df["date"].duplicated().any() or not df["date"].diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
            raise ValueError("Input dates must be unique, chronological and exactly hourly.")
    if len(df) != 96:
        raise ValueError(f"Input CSV must contain exactly 96 rows, got {len(df)}")
    return df[FEATURES].values

def main():
    # modify:audit-cli - import-safe CLI; consistent strict input validation.
    from agent.agent import OilTemperatureAgent
    ap = argparse.ArgumentParser(description="Industrial oil-temperature forecasting Agent")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--scaler", default=None)
    ap.add_argument("--config", default="agent/models/config.yaml")
    ap.add_argument("--input", default=None, help="CSV with exactly 96 hourly rows")
    ap.add_argument("--case", choices=["all", "stable", "slow_rise", "fast_rise"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.input and args.case:
        ap.error("Use either --input or --case, not both.")
    root=Path(__file__).resolve().parent
    if args.case == "all" or (args.case is None and args.input is None):
        agent=OilTemperatureAgent(args.checkpoint,args.scaler,args.config,args.device)
        import runpy
        runpy.run_path(str(root / "tests/run_agent_cases.py"))["main"](agent=agent)
        return
    input_path=Path(args.input) if args.input else root / "tests/cases" / args.case / "input.csv"
    if not input_path.is_absolute(): input_path=root/input_path
    history=load_history(input_path)
    agent=OilTemperatureAgent(args.checkpoint,args.scaler,args.config,args.device)
    print(json.dumps(agent.run(history),indent=2,ensure_ascii=False))

if __name__ == "__main__": main()
