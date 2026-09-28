"""modify:rag-evaluation. Fixed 12 queries; raw rank and thresholded metrics separated."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import yaml
import numpy as np
from agent.rag_tool import RAGTool

def evaluate(backend):
    cfg=yaml.safe_load((ROOT/'agent/models/config.yaml').read_text())['rag'].copy()
    cfg['backend']=backend
    cfg['similarity_threshold']=cfg['tfidf_threshold'] if backend=='tfidf' else cfg['similarity_threshold']
    rag=RAGTool(cfg); records=[]
    for q in json.loads((ROOT/'tests/rag_queries.json').read_text()):
        hits=rag.retrieve(q['query']); relevant=set(q['relevant_ids'])
        for hit in hits:
            hit['document_id']=hit['id'].rsplit('-C',1)[0]
            hit['relevant']=hit['document_id'] in relevant
        ranks=[i+1 for i,x in enumerate(hits) if x['relevant']]
        accepted={x['document_id'] for x in hits if x['accepted']}
        records.append({**q,'top3':hits,'refused':not accepted,
          'top1_correct':bool(hits and hits[0]['accepted'] and hits[0]['relevant']),
          'hit_at_3':bool(ranks),'reciprocal_rank_at_3':1/min(ranks) if ranks else 0,
          'accepted_hit_at_3':bool(accepted & relevant),
          'all_relevant_accepted':bool(relevant and relevant <= accepted)})
    answerable=[r for r in records if r['relevant_ids']]; unknown=[r for r in records if not r['relevant_ids']]
    multi=[r for r in answerable if r['category']=='multi']
    mean=lambda key,rs:float(np.mean([r[key] for r in rs]))
    metrics={'answerable_n':len(answerable),'unanswerable_n':len(unknown),
      'retrieval_accuracy_top1_thresholded':mean('top1_correct',answerable),
      'Hit@3_raw':mean('hit_at_3',answerable),'MRR@3_raw':mean('reciprocal_rank_at_3',answerable),
      'Hit@3_thresholded':mean('accepted_hit_at_3',answerable),
      'multi_all_evidence_rate':mean('all_relevant_accepted',multi),
      'no_answer_rejection_rate':mean('refused',unknown),
      'no_answer_false_accepts':sum(not r['refused'] for r in unknown),
      'answerable_false_rejections':sum(r['refused'] for r in answerable)}
    return {'backend':backend,'config':cfg,'metrics':metrics,'records':records}

def main():
    out=ROOT/'results/final';out.mkdir(parents=True,exist_ok=True)
    results={backend:evaluate(backend) for backend in ('tfidf','embedding')}
    (out/'rag_evaluation.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v['metrics'] for k,v in results.items()},indent=2));return results
if __name__=='__main__':main()
