import argparse, json
from pathlib import Path
import pandas as pd
from agent.agent import OilTemperatureAgent

FEATURES = ["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]

def load_history(csv_path):
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="raise")
        if not df["date"].is_monotonic_increasing:
            raise ValueError("Input dates must be chronological.")
    if len(df) != 96:
        raise ValueError(f"Input CSV must contain exactly 96 rows, got {len(df)}")
    return df[FEATURES].values

ap = argparse.ArgumentParser(description="Industrial oil-temperature forecasting Agent")
ap.add_argument("--checkpoint", default=None)  # modify:model-unification - use config by default.
ap.add_argument("--scaler", default=None)  # modify:model-unification
ap.add_argument("--config", default="agent/models/config.yaml")
ap.add_argument("--input", default=None,  # modify:run-all-cases - no input means all demo cases.
                help="CSV containing exactly the 96-hour history; if it has >96 rows, the last 96 are used.")
ap.add_argument("--case", default=None, help="Optional test case name under tests/cases/")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()
agent = OilTemperatureAgent(args.checkpoint, args.scaler, args.config, args.device)  # modify:run-all-cases
if args.case == "all" or (args.case is None and args.input is None):  # modify:run-all-cases
    import runpy  # modify:run-all-cases - reuse the existing runner without a new module.
    runner = runpy.run_path(str(Path(__file__).resolve().parent / "tests/run_agent_cases.py"))  # modify:run-all-cases
    runner["main"](agent=agent)  # modify:run-all-cases
    raise SystemExit(0)  # modify:run-all-cases

input_path = Path(args.input or "data/ETTh1.csv")  # modify:run-all-cases
if args.case:
    input_path = Path("tests/cases") / args.case / "input.csv"
if not input_path.is_absolute():  # modify:portable-paths
    input_path = Path(__file__).resolve().parent / input_path  # modify:portable-paths

df = pd.read_csv(input_path)
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Input CSV missing columns: {missing}")
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if len(df) >= 2 and not df["date"].is_monotonic_increasing:
        raise ValueError("Input dates must be chronological.")
history = df.tail(96)[FEATURES].values

result = agent.run(history)
print(json.dumps(result, indent=2, ensure_ascii=False))
