"""modify:closeout-tests. Verify provenance, metric definitions and final-source consistency."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import yaml
from agent.rag_tool import DEFAULT_KNOWLEDGE,RAGTool
from agent.risk_tool import load_frozen_thresholds
from metrics import regression_metrics

def main():
    d=json.loads((ROOT/'results/final/experiment.json').read_text())
    source_map=json.loads((ROOT/'results/final/source_snapshot.json').read_text())
    for rel,h in {**d['input_sha256'],**d['checkpoint_sha256']}.items():
        assert hashlib.sha256((ROOT/source_map.get(rel,rel)).read_bytes()).hexdigest()==h,rel
    # modify:audit-provenance - source snapshot preserves experiment history; runtime tested separately.
    assert (ROOT/'agent/models/best_model.pt').read_bytes()==(ROOT/'outputs/checkpoints/LSTM_best.pt').read_bytes()
    assert len(DEFAULT_KNOWLEDGE)==25 and len({x['id'] for x in DEFAULT_KNOWLEDGE})==25
    cfg=yaml.safe_load((ROOT/'agent/models/config.yaml').read_text());frozen=load_frozen_thresholds(cfg)
    assert frozen==d['risk_calibration']['thresholds']
    assert not d['risk_calibration']['case_inputs_accessed']
    # Raw predictions independently reproduce every published model metric.
    with np.load(ROOT/'results/final/predictions.npz') as z:
        for r in d['metrics']:
            for k,v in regression_metrics(z['truth'],z[r['Model']]).items():np.testing.assert_allclose(v,r[k],rtol=1e-10)
    gold=json.loads((ROOT/'tests/rag_queries.json').read_text());assert len(gold)==12
    assert all(sum(q['category']==c for q in gold)==3 for c in ['direct','paraphrase','multi','unanswerable'])
    ids={k['id'] for k in DEFAULT_KNOWLEDGE}
    for q in gold:assert set(q['relevant_ids'])<=ids
    for b,ev in d['rag'].items():
        rr=[];hits=[];top1=[];rejected=0
        for q in ev['records']:
            assert len(q['top3'])==3
            assert [x['score'] for x in q['top3']]==sorted([x['score'] for x in q['top3']],reverse=True)
            assert all(x['accepted']==(x['score']>=ev['config']['similarity_threshold']) for x in q['top3'])
            rel=[i+1 for i,x in enumerate(q['top3']) if x['document_id'] in q['relevant_ids']]
            if q['relevant_ids']:
                rr.append(1/min(rel) if rel else 0);hits.append(bool(rel));top1.append(bool(rel and min(rel)==1 and q['top3'][0]['accepted']))
            else:rejected+=not any(x['accepted'] for x in q['top3'])
        for k,v in [('MRR@3_raw',np.mean(rr)),('Hit@3_raw',np.mean(hits)),('retrieval_accuracy_top1_thresholded',np.mean(top1)),('no_answer_rejection_rate',rejected/3)]:np.testing.assert_allclose(v,ev['metrics'][k])
    # Chunk overlap must work for a document longer than current short KB records.
    r=RAGTool({**cfg['rag'],'backend':'tfidf'});chunks=r._chunk_documents([{'id':'test','text':' '.join('w'+str(i) for i in range(75))}]);assert len(chunks)==3
    assert chunks[0]['text'].split()[-8:]==chunks[1]['text'].split()[:8]
    cases=json.loads((ROOT/'output/all_case_results.json').read_text())
    for name,c in d['cases'].items():
        assert cases[name]['risk_level']==c['risk_level']
        np.testing.assert_allclose(cases[name]['forecast_output'],c['forecast_output'],rtol=1e-4,atol=1e-4)
        assert c['risk_intermediate_values']['thresholds']==frozen
    for rel in ['outputs/metrics.csv','results/model_comparison.csv','paper/tables/Table_Model_Comparison.csv']:
        df=pd.read_csv(ROOT/rel)
        np.testing.assert_allclose(df['MAE'],[r['MAE'] for r in d['metrics']])
    manifest=json.loads((ROOT/'results/final/generated_manifest.json').read_text())
    assert hashlib.sha256((ROOT/'results/final/experiment.json').read_bytes()).hexdigest()==manifest['experiment_sha256']
    for rel,h in manifest['artifacts'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
    print('PASS: provenance, checkpoint identity, frozen thresholds, predictions, RAG metrics, chunk overlap, cases, tables and generated files')
if __name__=='__main__':main()
