import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
os.makedirs('data', exist_ok=True)
N = 1500
start = datetime(2020,1,1)
dates = [ (start + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S') for i in range(N) ]
data = {}
data['date'] = dates
np.random.seed(0)
data['OT'] = np.random.randn(N).cumsum() + 100
for i in range(1,7):
    data[f'f{i}'] = np.random.randn(N).cumsum() + i*10
df = pd.DataFrame(data)
df.to_csv('data/ETTh1.csv', index=False)
print('Wrote synthetic data to data/ETTh1.csv, rows=', len(df))
