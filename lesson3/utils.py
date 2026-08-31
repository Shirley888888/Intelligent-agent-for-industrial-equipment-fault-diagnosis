"""数据层：本地读取 ETTh1.csv、z-score 归一化、滑窗构造样本、时间序列划分。

数据说明（ETTh1，电力变压器油温数据集）：
- 7 个变量：HUFL/HULL/MUFL/MULL/LUFL/LULL 六个电力负荷 + OT 油温
- 任务：用过去 seq_len 小时的全部 7 个变量，预测未来 pred_len 小时的 OT 油温
- 划分方式（修正版）：先按时间顺序把**原始序列**切成 Train/Validation/Test
  三段（默认 70/15/15），再在**各段内部**分别构造滑动窗口。这样每段内的
  样本互不跨越段边界，训练集与验证集边界样本不再高度重叠。

字段校验（修正版）：7 个标准特征列必须齐全，缺失直接抛错；数据必须非空、
不含 NaN / Inf。发现异常即中止，避免错误数据被自动放行。
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ETTH1_FEATURES = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
TARGET_COL = 'OT'


def _strict_validate(df):
    """严格校验 ETTh1 数据字段，异常直接抛错。"""
    # 1) 标准特征列必须齐全
    missing = [c for c in ETTH1_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f'ETTh1 数据缺少标准列 {missing}，实际列：{list(df.columns)}。'
            f'请使用真实 ETTh1 数据（需含 {ETTH1_FEATURES} 7 列）。'
        )
    # 2) 数据非空
    if df.empty:
        raise ValueError('ETTh1 数据为空，无法训练。')
    # 3) 无 NaN / Inf
    if df[ETTH1_FEATURES].isna().any().any():
        nan_cols = df.columns[df[ETTH1_FEATURES].isna().any()].tolist()
        raise ValueError(f'ETTh1 数据含 NaN（列：{nan_cols}），请先清洗数据。')
    num = df[ETTH1_FEATURES].values.astype(np.float64)
    if not np.isfinite(num).all():
        raise ValueError('ETTh1 数据含 Inf/非有限数值，请先清洗数据。')
    # 4) 若有时间列，要求严格递增（按时间顺序）
    if 'date' in df.columns:
        d = df['date'].values
        if np.any(d[1:] == d[:-1]):
            raise ValueError('ETTh1 数据含重复时间戳，请先按时间清理。')


def _build_windows(sub_scaled, target_idx, seq_len, pred_len):
    """在某一段已归一化序列内部构造滑窗样本。"""
    X, y = [], []
    L = len(sub_scaled)
    for i in range(L - seq_len - pred_len + 1):
        X.append(sub_scaled[i:i + seq_len])
        y.append(sub_scaled[i + seq_len:i + seq_len + pred_len, target_idx])
    if not X:
        raise ValueError(
            f'某段序列长度 {L} 不足以构造滑窗（需 >= seq_len+pred_len={seq_len + pred_len}）。'
            '请减小划分比例或提供更长数据。'
        )
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def load_etth1(csv_path, seq_len=96, pred_len=24, train_frac=0.70, val_frac=0.15, seed=42):
    df = pd.read_csv(csv_path)
    _strict_validate(df)

    cols = ETTH1_FEATURES  # 固定顺序；列缺失已在上方严格校验中拦截
    target_idx = cols.index(TARGET_COL)
    arr = df[cols].values.astype(np.float32)

    # 1) 先按时间顺序切分原始序列：train / val / test（不重叠、不泄漏）
    n_total = len(arr)
    n_tr = int(n_total * train_frac)
    n_va = int(n_total * val_frac)
    if n_tr < seq_len + pred_len:
        raise ValueError(f'训练段长度 {n_tr} 不足以构造滑窗（需 >= {seq_len + pred_len}）。')
    train_arr = arr[:n_tr]
    val_arr = arr[n_tr:n_tr + n_va]
    test_arr = arr[n_tr + n_va:]

    # 2) 只用训练段拟合归一化参数，避免泄漏
    scaler = StandardScaler().fit(train_arr)

    # 3) 在各段内部分别构造滑窗（段与段之间互不跨越）
    X_tr, y_tr = _build_windows(scaler.transform(train_arr).astype(np.float32),
                                target_idx, seq_len, pred_len)
    X_va, y_va = _build_windows(scaler.transform(val_arr).astype(np.float32),
                                target_idx, seq_len, pred_len)
    X_te, y_te = _build_windows(scaler.transform(test_arr).astype(np.float32),
                                target_idx, seq_len, pred_len)

    # 4) 打乱训练集顺序（固定种子，可复现）
    rng = np.random.default_rng(seed)
    i_tr = rng.permutation(len(X_tr))

    return {
        'X_train': X_tr[i_tr], 'y_train': y_tr[i_tr],
        'X_val': X_va, 'y_val': y_va,
        'X_test': X_te, 'y_test': y_te,
        'scaler': scaler, 'target_idx': target_idx, 'features': cols,
        'split_mode': 'time-first',  # 先切原始序列、再各段内滑窗
        'n_timesteps': {'train': len(train_arr), 'val': len(val_arr), 'test': len(test_arr)},
        'n_windows': {'train': len(X_tr), 'val': len(X_va), 'test': len(X_te)},
    }


def to_original_ot(arr, scaler, target_idx):
    """把归一化后的 OT 预测值还原为原始单位（摄氏度量级）。"""
    return arr * scaler.scale_[target_idx] + scaler.mean_[target_idx]
