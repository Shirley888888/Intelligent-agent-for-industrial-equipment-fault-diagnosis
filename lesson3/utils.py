"""数据层：本地读取 ETTh1.csv、z-score 归一化、滑窗构造样本、时间序列划分。

数据说明（ETTh1，电力变压器油温数据集）：
- 7 个变量：HUFL/HULL/MUFL/MULL/LUFL/LULL 六个电力负荷 + OT 油温
- 任务：用过去 seq_len 小时的全部 7 个变量，预测未来 pred_len 小时的 OT 油温
- 划分：严格按时间顺序 70/15/15，避免未来信息泄漏
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ETTH1_FEATURES = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
TARGET_COL = 'OT'


def load_etth1(csv_path, seq_len=96, pred_len=24, train_frac=0.70, val_frac=0.15, seed=42):
    df = pd.read_csv(csv_path)
    # 取 7 个标准特征列；若列名缺失则回退为所有数值列
    cols = [c for c in ETTH1_FEATURES if c in df.columns]
    if len(cols) < 2:
        cols = [c for c in df.columns if df[c].dtype != object and c != 'date']
    if TARGET_COL in cols:
        target_idx = cols.index(TARGET_COL)
    else:
        target_idx = len(cols) - 1
        print(f"[utils] 未找到列 {TARGET_COL}，将最后一列（idx={target_idx}）作为目标。")

    arr = df[cols].values.astype(np.float32)

    # 只用训练段拟合归一化参数，避免泄漏
    n_fit = int(len(arr) * train_frac)
    scaler = StandardScaler().fit(arr[:n_fit])
    scaled = scaler.transform(arr).astype(np.float32)

    # 滑窗构造样本 (seq_len, n_features) -> (pred_len,)
    X, y = [], []
    L = len(scaled)
    for i in range(L - seq_len - pred_len + 1):
        X.append(scaled[i:i + seq_len])
        y.append(scaled[i + seq_len:i + seq_len + pred_len, target_idx])
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    # 时间序列划分 70/15/15
    n_total = X.shape[0]
    n_tr = int(n_total * train_frac)
    n_va = int(n_total * val_frac)
    idx = np.arange(n_total)
    i_tr, i_va, i_te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    # 打乱训练集顺序（固定种子，可复现）
    rng = np.random.default_rng(seed)
    i_tr = rng.permutation(i_tr)

    return {
        'X_train': X[i_tr], 'y_train': y[i_tr],
        'X_val': X[i_va], 'y_val': y[i_va],
        'X_test': X[i_te], 'y_test': y[i_te],
        'scaler': scaler, 'target_idx': target_idx, 'features': cols,
    }


def to_original_ot(arr, scaler, target_idx):
    """把归一化后的 OT 预测值还原为原始单位（摄氏度量级）。"""
    return arr * scaler.scale_[target_idx] + scaler.mean_[target_idx]
