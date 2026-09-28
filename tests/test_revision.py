"""modify:audit-regression. Behavioral checks for review fixes, no score tuning."""
import sys,json,tempfile,importlib.util,time,statistics,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import yaml
from agent.rag_tool import RAGTool
from agent.risk_tool import calculate_dynamic_thresholds,load_frozen_thresholds
from run_agent import load_history
from experiment_paths import reproduction_dir

def raises(fn,kind=ValueError):
    try:fn()
    except kind:return
    raise AssertionError('Expected exception not raised')

def main():
    cfg=yaml.safe_load((ROOT/'agent/models/config.yaml').read_text()); rc=cfg['risk']
    r=RAGTool({**cfg['rag'],'backend':'tfidf'})
    assert r.threshold==cfg['rag']['tfidf_threshold']
    raises(lambda:r.retrieve('  '));raises(lambda:r.retrieve_many(['good','']))
    assert r.retrieve_many([])==[]
    raises(lambda:RAGTool({**cfg['rag'],'backend':'tfidf','top_k':0}))
    raises(lambda:RAGTool({**cfg['rag'],'backend':'tfidf','tfidf_threshold':float('nan')}))
    for dest in ['results/final','results/final/nested','results','paper','archive']:
        raises(lambda:reproduction_dir('test',ROOT/dest))
    df=pd.read_csv(ROOT/'tests/cases/stable/input.csv')
    with tempfile.TemporaryDirectory() as temp:
        p=Path(temp)/'input.csv';df.to_csv(p,index=False);assert load_history(p).shape==(96,7)
        missing_date=df.drop(columns=['date']);missing_date.to_csv(p,index=False);raises(lambda:load_history(p))
        for mode in ['duplicate','gap','short']:
            changed=df.copy()
            if mode=='duplicate':changed.loc[2,'date']=changed.loc[1,'date']
            if mode=='gap':changed.loc[2,'date']='2030-01-01 00:00:00'
            if mode=='short':changed=changed.iloc[:-1]
            changed.to_csv(p,index=False);raises(lambda:load_history(p))
        # Stored numeric thresholds cannot be altered while keeping data/config hashes.
        frozen=json.loads((ROOT/rc['frozen_path']).read_text());frozen['thresholds']['temperature']['medium']-=1
        q=Path(temp)/'altered.json';q.write_text(json.dumps(frozen))
        changed={**cfg,'risk':{**rc,'frozen_path':str(q)}};raises(lambda:load_frozen_thresholds(changed))
    spec=importlib.util.spec_from_file_location('oldrisk',ROOT/'archive/closeout_v1_source/agent/risk_tool.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    args=(ROOT/rc['data_path'],cfg['target'],rc['train_ratio'],rc['forecast_horizon'],rc['quantiles'])
    assert calculate_dynamic_thresholds(*args)==old.calculate_dynamic_thresholds(*args)
    elapsed={}
    for name,fn in [('loop',old.calculate_dynamic_thresholds),('vectorized',calculate_dynamic_thresholds)]:
        ts=[]
        for _ in range(5):
            start=time.perf_counter();fn(*args);ts.append(time.perf_counter()-start)
        elapsed[name+'_median_s']=statistics.median(ts)
    queries=[q['query'] for q in json.loads((ROOT/'tests/rag_queries.json').read_text())]
    rag=RAGTool(cfg['rag'])
    singles=[rag.retrieve(q) for q in queries];batch=rag.retrieve_many(queries)
    maxdiff=0
    for a,b in zip(singles,batch):
        assert [x['id'] for x in a]==[x['id'] for x in b]
        assert [x['accepted'] for x in a]==[x['accepted'] for x in b]
        diff=np.max(np.abs(np.array([x['score'] for x in a])-np.array([x['score'] for x in b])))
        maxdiff=max(maxdiff,float(diff));assert diff<1e-5
    for name,fn in [('sequential',lambda:[rag.retrieve(q) for q in queries]),('batch',lambda:rag.retrieve_many(queries))]:
        ts=[]
        for _ in range(5):
            start=time.perf_counter();fn();ts.append(time.perf_counter()-start)
        elapsed[name+'_median_s']=statistics.median(ts)
    out={'checks':'PASS','quantile_loop_vs_vectorized_exact':True,'rag_rank_and_acceptance_equal':True,
      'max_batch_score_abs_difference':maxdiff,'timing':elapsed,'method':'CPU, two threads, 5 warm repetitions, median. RAG excludes model loading; risk includes CSV loading.'}
    (ROOT/'results/audit/revision_checks.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
