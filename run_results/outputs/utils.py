import os
import pandas as pd
import numpy as np

ETTH1_RAW_URL = 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/master/ETT/ETTh1.csv'

def download_etth1(dest="data/ETTh1.csv"):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"Found existing {dest}")
        return dest
    try:
        df = pd.read_csv(ETTH1_RAW_URL)
        df.to_csv(dest, index=False)
        print(f"Downloaded ETTh1 to {dest}")
        return dest
    except Exception as e:
        print("Failed to download ETTh1:", e)
        raise

class SlidingWindowDataset:
    def __init__(self, csv_path, input_len=96, pred_len=24, features=None, target='OT'):
        self.df = pd.read_csv(csv_path)
        # try common ETT columns; if not found, pick numeric columns
        if features is None:
            cols = [c for c in self.df.columns if self.df[c].dtype!=object and c!='date']
            if target in cols:
                # remove target from features
                cols = [c for c in cols if c!=target]
            # choose first up to 7 features
            features = cols[:7]
        self.features = features
        self.target = target
        self.input_len = input_len
        self.pred_len = pred_len
        self.arr = self.df[self.features + [self.target]].dropna().values

    def __len__(self):
        L = len(self.arr)
        return max(0, L - self.input_len - self.pred_len + 1)

    def __getitem__(self, idx):
        x = self.arr[idx:idx+self.input_len, :len(self.features)]
        y = self.arr[idx+self.input_len:idx+self.input_len+self.pred_len, -1]
        return x.astype('float32'), y.astype('float32')

    def split_timebased(self, train_frac=0.7, val_frac=0.15):
        N = len(self)
        n_train = int(N * train_frac)
        n_val = int(N * val_frac)
        train_idx = list(range(0, n_train))
        val_idx = list(range(n_train, n_train + n_val))
        test_idx = list(range(n_train + n_val, N))
        return train_idx, val_idx, test_idx
