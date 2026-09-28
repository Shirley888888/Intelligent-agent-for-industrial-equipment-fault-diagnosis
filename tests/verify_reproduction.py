"""Compare real reruns to the immutable experiment without promoting new scores."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np

def main():
    d=json.loads((ROOT/'results/final/experiment.json').read_text())
    r=json.loads((ROOT/'results/reruns/review_forecast/forecast_evaluation.json').read_text())
    diffs=[]
    for a,b in zip(d['metrics'],r['metrics']):
        assert a['Model']==b['Model']
        for k in ['MAE','MSE','MeanTempError','MaxTempError','MaxRiseRateError','PeakError']:
            delta=abs(a[k]-b[k]);assert delta<1e-5,(a['Model'],k,delta)
            diffs.append({'Model':a['Model'],'Metric':k,'abs_difference':delta})
    r=json.loads((ROOT/'results/reruns/review_rag/rag_evaluation.json').read_text());maxdiff=0
    for backend,ref in d['rag'].items():
        assert ref['metrics']==r[backend]['metrics']
        for a,b in zip(ref['records'],r[backend]['records']):
            assert a['id']==b['id'] and a['refused']==b['refused']
            assert [h['id'] for h in a['top3']]==[h['id'] for h in b['top3']]
            assert [h['accepted'] for h in a['top3']]==[h['accepted'] for h in b['top3']]
            for x,y in zip(a['top3'],b['top3']):maxdiff=max(maxdiff,abs(x['score']-y['score']))
    assert maxdiff<1e-5
    with np.load(ROOT/'results/final/predictions.npz') as z,np.load(ROOT/'results/reruns/review_forecast/predictions.npz') as v:
        pdiff={k:float(np.max(np.abs(z[k]-v[k]))) for k in z.files}
        for k in z.files:np.testing.assert_allclose(z[k],v[k],atol=1e-5,rtol=1e-6)
    # Public implementation hashes are separate from the original experiment's source hashes.
    manifest_path=ROOT/'results/audit/implementation_manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        for rel,h in manifest['source_sha256'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
    result={'status':'PASS','formal_experiment_id':d['experiment_id'],'forecast_metric_differences':diffs,'prediction_max_abs_differences':pdiff,'rag_metrics_equal':True,'rag_ranks_and_acceptance_equal':True,'rag_max_score_abs_difference':maxdiff}
    (ROOT/'results/audit/reproduction_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('PASS: frozen vs rerun predictions, metrics, all RAG ranks and acceptance; implementation hashes')
if __name__=='__main__':main()
