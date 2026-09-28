"""modify:single-experiment. Evaluate fixed packaged checkpoints, without training or selection."""
from pathlib import Path
import json,hashlib,platform
import numpy as np
import pandas as pd
import torch,yaml
from data import load_splits
from models import MODEL_CLASSES
from metrics import regression_metrics
ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    cfg=yaml.safe_load((ROOT/'configs/experiment.yaml').read_text())
    df,raw,splits,scaler=load_splits(ROOT/'data/ETTh1.csv',cfg['features'],cfg['target'],cfg['split'],96,24,1)
    X,y=splits[2];ti=cfg['features'].index('OT'); truth=y.astype('float64')*scaler.scale_[ti]+scaler.mean_[ti]
    pred=np.repeat(X[:,-1,ti:ti+1].astype('float64'),24,axis=1)*scaler.scale_[ti]+scaler.mean_[ti]
    rows=[{'Model':'Persistence',**regression_metrics(truth,pred),'BestEpoch':0}]; hashes={};preds={'Persistence':pred}
    for name in cfg['models']:
        path=ROOT/f'outputs/checkpoints/{name}_best.pt';c=torch.load(path,map_location='cpu',weights_only=False)
        assert c['model_name']==name
        assert c['config']['features']==cfg['features'] and c['config']['split']==cfg['split']
        assert c['config']['input_len']==96 and c['config']['pred_len']==24
        np.testing.assert_allclose(c['scaler_mean'],scaler.mean_,rtol=1e-7)
        np.testing.assert_allclose(c['scaler_scale'],scaler.scale_,rtol=1e-7)
        model=MODEL_CLASSES[name](96,7,24,**c['model_config']);model.load_state_dict(c['model_state']);model.eval()
        with torch.no_grad(): pred=np.concatenate([model(torch.from_numpy(X[i:i+128])).numpy() for i in range(0,len(X),128)])
        pred=pred.astype('float64')*scaler.scale_[ti]+scaler.mean_[ti];preds[name]=pred
        rows.append({'Model':name,**regression_metrics(truth,pred),'BestEpoch':c['best_epoch'],'BestValLoss':c['best_val_loss']})
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        print(name,rows[-1]['MAE'],flush=True)
    out=ROOT/'results/final';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'predictions.npz',truth=truth,**preds)
    result={'experiment_id':'ETTh1_CLOSEOUT_20260929_v1','protocol':'Fixed existing four checkpoints; CPU float32 inference, float64 inverse scaling/metrics; no retraining or test-based model selection.',
     'dataset_sha256':hashlib.sha256((ROOT/'data/ETTh1.csv').read_bytes()).hexdigest(),
     'split_rows':[len(x) for x in raw],'split_windows':[len(x[0]) for x in splits],
     'deployment_model':'LSTM','checkpoint_sha256':hashes,'metrics':rows,
     'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__},
     'limitations':['Historical checkpoints; no new training run or confidence intervals.','Deployment LSTM retained by project design, not selected on final test scores.']}
    (out/'forecast_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
if __name__=='__main__':main()
