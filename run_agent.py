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
ap.add_argument("--checkpoint", default="agent/models/best_model.pt")
ap.add_argument("--scaler", default="agent/models/scaler.pkl")
ap.add_argument("--config", default="agent/models/config.yaml")
ap.add_argument("--input", default="data/ETTh1.csv",
                help="CSV containing exactly the 96-hour history; if it has >96 rows, the last 96 are used.")
ap.add_argument("--case", default=None, help="Optional test case name under tests/cases/")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

input_path = Path(args.input)
if args.case:
    input_path = Path("tests/cases") / args.case / "input.csv"

df = pd.read_csv(input_path)
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Input CSV missing columns: {missing}")
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if len(df) >= 2 and not df["date"].is_monotonic_increasing:
        raise ValueError("Input dates must be chronological.")
history = df.tail(96)[FEATURES].values

agent = OilTemperatureAgent(args.checkpoint, args.scaler, args.config, args.device)
result = agent.run(history)
print(json.dumps(result, indent=2, ensure_ascii=False))
