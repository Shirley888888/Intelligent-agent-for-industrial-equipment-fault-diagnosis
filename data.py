import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    def __init__(self,X,y):
        self.X=np.asarray(X,dtype=np.float32); self.y=np.asarray(y,dtype=np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.y[i]

def load_splits(path, features, target, split, input_len, pred_len, stride=1):
    df=pd.read_csv(path,parse_dates=["date"])
    if list(df.columns)!=["date"]+features:
        missing=[c for c in ["date"]+features if c not in df.columns]
        if missing: raise ValueError(f"Missing columns: {missing}")
    n=len(df); a=int(n*split[0]); b=a+int(n*split[1])
    raw=[df.iloc[:a].copy(),df.iloc[a:b].copy(),df.iloc[b:].copy()]
    scaler=StandardScaler().fit(raw[0][features].values)
    out=[]
    ti=features.index(target)
    for part in raw:
        z=scaler.transform(part[features].values)
        X=[]; y=[]
        last=len(z)-input_len-pred_len+1
        for s in range(0,max(0,last),stride):
            X.append(z[s:s+input_len])
            y.append(z[s+input_len:s+input_len+pred_len,ti])
        out.append((np.array(X,np.float32),np.array(y,np.float32)))
    return df, raw, out, scaler
