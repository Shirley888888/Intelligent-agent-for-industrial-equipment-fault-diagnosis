import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
import pandas as pd
from agent.agent import OilTemperatureAgent

FEATURES=["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]

def main():
    root=Path("tests/cases")
    agent=OilTemperatureAgent()
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        df=pd.read_csv(case_dir/"input.csv")
        history=df[FEATURES].tail(96).values
        result=agent.run(history)
        payload={
            "case":case_dir.name,
            "input_file":str(case_dir/"input.csv"),
            "prediction":result["forecast"],
            "risk":result["risk"],
            "model":result["model"],
            "input_rows":len(history)
        }
        with open(case_dir/"prediction.json","w",encoding="utf-8") as f:
            json.dump(payload,f,indent=2,ensure_ascii=False)
        print(json.dumps({
            "case":case_dir.name,
            "model":result["model"],
            "max":result["risk"]["max"],
            "rise":result["risk"]["rise"],
            "max_slope":result["risk"]["max_slope"],
            "risk_level":result["risk"]["risk_level"]
        },ensure_ascii=False))
if __name__=="__main__":
    main()
