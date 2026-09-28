"""modify:freeze-risk. Train calibration, validation audit, then freeze; never reads cases."""
from pathlib import Path
import json, hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yaml
from agent.risk_tool import calculate_dynamic_thresholds, analyze
ROOT=Path(__file__).resolve().parent

def main():
    cfg=yaml.safe_load((ROOT/'agent/models/config.yaml').read_text())
    r=cfg['risk']; data=ROOT/r['data_path']; out=ROOT/r['frozen_path']
    calibration_config={k:r[k] for k in ('train_ratio','forecast_horizon','quantiles')}
    calibration_config['target']=cfg['target']
    digest=hashlib.sha256(data.read_bytes()).hexdigest()
    if out.exists():
        old=json.loads(out.read_text())
        if old['calibration_config']!=calibration_config or old['dataset_sha256']!=digest:
            raise ValueError('Frozen calibration differs; create a new experiment version, not case-based retuning.')
        return old
    thresholds=calculate_dynamic_thresholds(data,cfg['target'],r['train_ratio'],r['forecast_horizon'],r['quantiles'])
    thresholds['source']['data_path']=r['data_path']
    df=pd.read_csv(data); a=int(len(df)*r['train_ratio']); b=a+int(len(df)*0.15)
    values=df[cfg['target']].to_numpy()[a:b]; h=r['forecast_horizon']
    levels=[analyze(values[i:i+h],thresholds=thresholds)['risk_level'] for i in range(len(values)-h+1)]
    artifact={'created_utc':datetime.now(timezone.utc).isoformat(),'dataset_sha256':digest,
      'calibration_config':calibration_config,'thresholds':thresholds,
      'protocol':'Predeclared q90/q95 for all three 24h indicators; train only; validation audit only; no demo tuning.',
      'validation':{'row_range_half_open':[a,b],'windows':len(levels),
        'level_counts':{k:levels.count(k) for k in ('LOW','MEDIUM','HIGH')}},
      'case_inputs_accessed':False}
    out.write_text(json.dumps(artifact,indent=2),encoding='utf-8')
    print(json.dumps(artifact,indent=2)); return artifact
if __name__=='__main__': main()
