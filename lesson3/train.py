"""训练主流程：config.yaml 驱动，重跑全部 5 个模型。

每轮运行：
  python train.py                 # 自动取下一轮编号 r1, r2, ...
  python train.py --round r3      # 手动指定轮次编号

输出（lesson3/outputs/）：
  - metrics.csv                   每轮全部模型的指标（可累积追溯）
  - figures/{round}_{model}_loss_curve.png    train/val loss 随 epoch 曲线
  - figures/{round}_{model}_pred_vs_true.png  测试集首个样本预测 vs 真实
  - figures/{round}_all_models_val_loss.png   5 模型 val loss 对比
"""
import argparse
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from models import LinearModel, MLP, Conv1DModel, LSTMModel
from utils import load_etth1, to_original_ot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
ROUND_FILE = os.path.join(OUTPUT_DIR, '.round')
MODELS = ['baseline', 'linear', 'mlp', 'cnn', 'lstm']


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def next_round(manual=None):
    if manual:
        return manual
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cur = 0
    if os.path.exists(ROUND_FILE):
        with open(ROUND_FILE) as f:
            cur = int(f.read().strip() or 0)
    nxt = cur + 1
    with open(ROUND_FILE, 'w') as f:
        f.write(str(nxt))
    return f"r{nxt}"


def build_model(name, cfg, n_features, seq_len, pred_len):
    m = cfg['models']
    if name == 'linear':
        return LinearModel(seq_len, n_features, pred_len)
    if name == 'mlp':
        return MLP(seq_len, n_features, hidden=m['mlp']['hidden'],
                   layers=m['mlp']['layers'], pred_len=pred_len)
    if name == 'cnn':
        return Conv1DModel(n_features, channels=m['cnn']['channels'],
                           kernel=m['cnn']['kernel'], pred_len=pred_len)
    if name == 'lstm':
        return LSTMModel(n_features, hidden=m['lstm']['hidden'],
                         layers=m['lstm']['layers'], pred_len=pred_len)
    raise ValueError(f'未知模型: {name}')


def structure_desc(name, cfg, n_features, seq_len, pred_len):
    m = cfg['models']
    if name == 'baseline':
        return 'Last Value（取输入最后1个OT值重复24步）'
    if name == 'linear':
        return f'Linear({seq_len * n_features}→{pred_len})'
    if name == 'mlp':
        return f'MLP({seq_len * n_features}→{m["mlp"]["hidden"]}×{m["mlp"]["layers"]}层→{pred_len})'
    if name == 'cnn':
        return f'CNN(kernel={m["cnn"]["kernel"]}, channels={m["cnn"]["channels"]})'
    if name == 'lstm':
        return f'LSTM({m["lstm"]["layers"]}层, hidden={m["lstm"]["hidden"]})'
    return name


def make_loader(X, y, batch_size, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, loader, optim, loss_fn, device):
    model.train()
    total, cnt = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optim.step()
        total += loss.item() * xb.size(0)
        cnt += xb.size(0)
    return total / cnt


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).cpu().numpy())
        trues.append(yb.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


@torch.no_grad()
def baseline_predict(loader, target_idx):
    preds, trues = [], []
    for xb, yb in loader:
        # 取输入窗口最后一个时刻的 OT（第 target_idx 列），重复 pred_len 步
        last = xb[:, -1, target_idx].numpy()[:, None]
        preds.append(np.repeat(last, yb.shape[1], axis=1))
        trues.append(yb.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def mae_mse(pred_orig, true_orig):
    return (mean_absolute_error(true_orig.flatten(), pred_orig.flatten()),
            mean_squared_error(true_orig.flatten(), pred_orig.flatten()))


def save_loss_curve(rnd, model_name, train_losses, val_losses, best_epoch):
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='train loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='val loss')
    plt.axvline(best_epoch, color='gray', ls='--', lw=0.8,
                label=f'best epoch {best_epoch}')
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.title(f'{rnd} | {model_name} loss curve')
    plt.legend()
    plt.grid(alpha=0.3)
    path = os.path.join(FIG_DIR, f'{rnd}_{model_name}_loss_curve.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return path


def save_pred_vs_true(rnd, model_name, pred_orig, true_orig):
    plt.figure(figsize=(7, 3))
    t0, p0 = true_orig[0], pred_orig[0]
    x = np.arange(len(t0))
    plt.plot(x, t0, 'o-', label='truth', ms=3)
    plt.plot(x, p0, 's--', label='pred', ms=3)
    plt.xlabel('hour offset')
    plt.ylabel('OT')
    plt.title(f'{rnd} | {model_name} test first sample')
    plt.legend()
    plt.grid(alpha=0.3)
    path = os.path.join(FIG_DIR, f'{rnd}_{model_name}_pred_vs_true.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return path


def save_all_val_curve(rnd, all_val_losses):
    plt.figure(figsize=(7, 4))
    for name, losses in all_val_losses.items():
        plt.plot(range(1, len(losses) + 1), losses, label=name)
    plt.xlabel('epoch')
    plt.ylabel('val MSE loss')
    plt.title(f'{rnd} | all models val loss')
    plt.legend()
    plt.grid(alpha=0.3)
    path = os.path.join(FIG_DIR, f'{rnd}_all_models_val_loss.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return path


def run(cfg, rnd):
    os.makedirs(FIG_DIR, exist_ok=True)
    data_cfg = cfg['data']
    tr_cfg = cfg['training']
    model_cfg = cfg['models']

    seq_len, pred_len = data_cfg['seq_len'], data_cfg['pred_len']
    epochs, batch_size, lr = tr_cfg['epochs'], tr_cfg['batch_size'], tr_cfg['lr']
    set_seed(tr_cfg.get('seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[round={rnd}] device={device}  '
          f'epochs={epochs} batch={batch_size} lr={lr} seq={seq_len}->pred={pred_len}')

    d = load_etth1(os.path.join(BASE_DIR, 'data', 'ETTh1.csv'),
                   seq_len=seq_len, pred_len=pred_len,
                   train_frac=data_cfg['train_frac'], val_frac=data_cfg['val_frac'],
                   seed=tr_cfg.get('seed', 42))
    n_features = len(d['features'])

    train_loader = make_loader(d['X_train'], d['y_train'], batch_size)
    val_loader = make_loader(d['X_val'], d['y_val'], batch_size)
    test_loader = make_loader(d['X_test'], d['y_test'], batch_size)
    loss_fn = nn.MSELoss()

    rows, all_val_losses = [], {}

    for name in MODELS:
        structure = structure_desc(name, cfg, n_features, seq_len, pred_len)
        print(f'\n===== {name}: {structure} =====')

        if name == 'baseline':
            t0 = time.time()
            pred, true = baseline_predict(test_loader, d['target_idx'])
            elapsed = time.time() - t0
            mae, mse = mae_mse(to_original_ot(pred, d['scaler'], d['target_idx']),
                               to_original_ot(true, d['scaler'], d['target_idx']))
            rows.append({'round': rnd, 'model': name, 'structure': structure,
                         'params': 0, 'mae': mae, 'mse': mse,
                         'train_time_s': round(elapsed, 2), 'best_epoch': '-',
                         'best_val_loss': '-', 'final_val_loss': '-',
                         'final_train_loss': '-'})
            save_pred_vs_true(rnd, name, pred, true)
            print(f'  MAE={mae:.4f}  MSE={mse:.4f}  time={elapsed:.2f}s  (baseline 无训练)')
            continue

        model = build_model(name, cfg, n_features, seq_len, pred_len).to(device)
        params = sum(p.numel() for p in model.parameters())
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        train_losses, val_losses = [], []
        best_val, best_state, best_epoch = float('inf'), None, 1
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            tr_loss = train_one_epoch(model, train_loader, optim, loss_fn, device)
            val_pred, val_true = predict(model, val_loader, device)
            val_loss = loss_fn(torch.from_numpy(val_pred),
                               torch.from_numpy(val_true)).item()
            train_losses.append(tr_loss)
            val_losses.append(val_loss)
            if val_loss < best_val:
                best_val, best_state, best_epoch = val_loss, model.state_dict(), epoch
            if epoch % 10 == 0 or epoch == epochs:
                print(f'  epoch {epoch:3d}/{epochs}  train_loss={tr_loss:.5f}  val_loss={val_loss:.5f}')

        elapsed = time.time() - t0
        model.load_state_dict(best_state)

        pred, true = predict(model, test_loader, device)
        mae, mse = mae_mse(to_original_ot(pred, d['scaler'], d['target_idx']),
                           to_original_ot(true, d['scaler'], d['target_idx']))
        rows.append({'round': rnd, 'model': name, 'structure': structure,
                     'params': params, 'mae': mae, 'mse': mse,
                     'train_time_s': round(elapsed, 2), 'best_epoch': best_epoch,
                     'best_val_loss': round(best_val, 6),
                     'final_val_loss': round(val_losses[-1], 6),
                     'final_train_loss': round(train_losses[-1], 6)})
        all_val_losses[name] = val_losses

        save_loss_curve(rnd, name, train_losses, val_losses, best_epoch)
        save_pred_vs_true(rnd, name, to_original_ot(pred, d['scaler'], d['target_idx']),
                          to_original_ot(true, d['scaler'], d['target_idx']))
        print(f'  params={params}  test MAE={mae:.4f}  MSE={mse:.4f}  '
              f'best_epoch={best_epoch}  time={elapsed:.1f}s')

    # 汇总指标：追加到 metrics.csv
    df_new = pd.DataFrame(rows)
    metrics_path = os.path.join(OUTPUT_DIR, 'metrics.csv')
    if os.path.exists(metrics_path):
        df_old = pd.read_csv(metrics_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(metrics_path, index=False, encoding='utf-8-sig')

    # 5 模型 val loss 对比图（baseline 无曲线，跳过）
    if all_val_losses:
        save_all_val_curve(rnd, all_val_losses)

    print('\n===== 本轮结果汇总 =====')
    print(df_new.to_string(index=False))
    print(f'\n指标已追加到 {metrics_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(BASE_DIR, 'config.yaml'))
    parser.add_argument('--round', default=None, help='轮次编号，如 r3；缺省自动累加')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    rnd = next_round(args.round)
    run(config, rnd)
