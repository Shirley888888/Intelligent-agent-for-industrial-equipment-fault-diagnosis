"""Minimal executable ATREA demonstration with synthetic, clearly labeled data."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timedelta, timezone
import numpy as np
from .orchestrator import run_if_due
from .store import ExperienceStore
from .trigger import TriggerConfig


def main():
    parser=argparse.ArgumentParser(description="Run a trigger-gated ATREA demo cycle")
    parser.add_argument("--workspace",default="agent_workspace/atrea_experience.sqlite3")
    parser.add_argument("--synthetic-demo",action="store_true",
                        help="Use fabricated errors/features and a fake paper result; not a real benchmark")
    args=parser.parse_args()
    if not args.synthetic_demo:
        parser.error("This CLI currently runs only with --synthetic-demo; production data must be supplied by the host agent.")
    store=ExperienceStore(args.workspace)
    errors=np.r_[np.ones(40),np.ones(50)*1.4]
    rng=np.random.default_rng(17)
    reference=rng.normal(0,1,(200,2)); recent=rng.normal(0.8,1,(100,2))
    def fake_search(query,max_results=15,submitted_after=None):
        return [{"paper_id":"demo-2026-001","title":"Synthetic paper discovery fixture",
                 "abstract":"Fabricated fixture describing patch representations.",
                 "url":"demo://paper","published":"2026-01-01","source":"synthetic_demo",
                 "extracted_methods":["patching"]}]
    def runner(proposal,batch_id):
        records=[]
        for model,mae in (("Linear-baseline",1.0),("PatchTST-demo",0.96)):
            for seed in (11,22,33):
                records.append({"dataset":"synthetic-demo","split_hash":"demo-frozen-split-v1",
                    "model_name":model,"model_version":"demo","config":{"context":96},
                    "hyperparameters":{"seed":seed},"seed":seed,"code_version":"demo-only",
                    "metrics":{"mae":mae},"is_candidate":model!="Linear-baseline"})
        return records
    result=run_if_due(store=store,errors=errors,reference_features=reference,
        recent_features=recent,last_train_time=datetime.now(timezone.utc)-timedelta(days=30),
        experiment_runner=runner,config=TriggerConfig(),paper_search=fake_search,
        candidate_gate=lambda candidate,all_rows:{"allow_upgrade":False,"status":"DEMO_NO_PROMOTION"})
    print(json.dumps(result["trigger"],ensure_ascii=False,indent=2))
    if result["cycle"]: print(result["cycle"]["markdown"])


if __name__=="__main__": main()
