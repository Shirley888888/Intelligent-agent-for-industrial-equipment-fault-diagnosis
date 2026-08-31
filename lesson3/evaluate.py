"""最终方案在 Test 集上的正式评价（修正版流程的最后一步）。

用法（在模型结构/超参通过 Validation 确定之后运行一次）：
  python evaluate.py --model linear --model lstm
  python evaluate.py --model lstm --epochs 2      # 快速验证流程

说明：
  - 训练仍使用 Validation 选择 best epoch（不接触 Test）；
  - 选定模型后仅在 Test 集上做**一次**正式预测与评价；
  - 输出 outputs/final_test_results.csv 与 test 预测图，
    该结果不再用于任何调参，作为最终方案的正式证据。

输出（outputs/）：
  - final_test_results.csv    round, model, structure, params, test_mae, test_mse, ...
  - figures/final_{model}_pred_vs_true.png    Test 首个样本预测 vs 真实（反归一化）
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

from train import (set_seed, build_model, structure_desc, make_loader,
                   train_one_epoch, predict, baseline_predict, mae_mse)
from utils import load_etth1, to_original_ot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')


def save_test_pred_vs_true(tag, model_name, pred_orig, true_orig):
    plt.figure(figsize=(7, 3))
    t0, p0 = true_orig[0], pred_orig[0]
    x = np.arange(len(t0))
    plt.plot(x, t0, 'o-', label='truth', ms=3)
    plt.plot(x, p0, 's--', label='pred', ms=3)
    plt.xlabel('hour offset')
    plt.ylabel('OT')
    plt.title(f'{tag} | {model_name} test first sample (final evaluation)')
    plt.legend()
    plt.grid(alpha=0.3)
    path = os.path.join(FIG_DIR, f'final_{model_name}_pred_vs_true.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return path


def run(cfg, tag, model_names):
    os.makedirs(FIG_DIR, exist_ok=True)
    data_cfg = cfg['data']
    tr_cfg = cfg['training']

    seq_len, pred_len = data_cfg['seq_len'], data_cfg['pred_len']
    epochs, batch_size, lr = tr_cfg['epochs'], tr_cfg['batch_size'], tr_cfg['lr']
    set_seed(tr_cfg.get('seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[final-eval tag={tag}] device={device}  '
          f'epochs={epochs} batch={batch_size} lr={lr} seq={seq_len}->pred={pred_len}')

    d = load_etth1(os.path.join(BASE_DIR, 'data', 'ETTh1.csv'),
                   seq_len=seq_len, pred_len=pred_len,
                   train_frac=data_cfg['train_frac'], val_frac=data_cfg['val_frac'],
                   seed=tr_cfg.get('seed', 42))
    n_features = len(d['features'])
    print(f'[final-eval] split={d["split_mode"]}  '
          f'windows: train={d["n_windows"]["train"]} '
          f'val={d["n_windows"]["val"]} test={d["n_windows"]["test"]}')

    train_loader = make_loader(d['X_train'], d['y_train'], batch_size)
    val_loader = make_loader(d['X_val'], d['y_val'], batch_size)
    test_loader = make_loader(d['X_test'], d['y_test'], batch_size)
    loss_fn = nn.MSELoss()

    rows = []
    for name in model_names:
        structure = structure_desc(name, cfg, n_features, seq_len, pred_len)
        print(f'\n===== FINAL EVAL | {name}: {structure} =====')

        if name == 'baseline':
            t0 = time.time()
            pred, true = baseline_predict(test_loader, d['target_idx'])
            elapsed = time.time() - t0
            pred_orig = to_original_ot(pred, d['scaler'], d['target_idx'])
            true_orig = to_original_ot(true, d['scaler'], d['target_idx'])
            test_mae, test_mse = mae_mse(pred_orig, true_orig)
            rows.append({'round': tag, 'model': name, 'structure': structure,
                         'params': 0, 'test_mae': test_mae, 'test_mse': test_mse,
                         'train_time_s': round(elapsed, 2), 'best_epoch': '-',
                         'best_val_loss': '-', 'split_mode': 'time-first'})
            save_test_pred_vs_true(tag, name, pred_orig, true_orig)
            print(f'  test MAE={test_mae:.4f}  test MSE={test_mse:.4f}  '
                  f'time={elapsed:.2f}s  (baseline 无训练)')
            continue

        set_seed(tr_cfg.get('seed', 42))
        model = build_model(name, cfg, n_features, seq_len, pred_len).to(device)
        params = sum(p.numel() for p in model.parameters())
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        best_val, best_state, best_epoch = float('inf'), None, 1
        t0 = time.time()
        for epoch in range(1, epochs + 1):
            tr_loss = train_one_epoch(model, train_loader, optim, loss_fn, device)
            val_pred, val_true = predict(model, val_loader, device)
            val_loss = loss_fn(torch.from_numpy(val_pred),
                               torch.from_numpy(val_true)).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
        elapsed = time.time() - t0
        model.load_state_dict(best_state)

        # 最终方案：在 Test 集上做一次正式预测
        pred, true = predict(model, test_loader, device)
        pred_orig = to_original_ot(pred, d['scaler'], d['target_idx'])
        true_orig = to_original_ot(true, d['scaler'], d['target_idx'])
        test_mae, test_mse = mae_mse(pred_orig, true_orig)
        rows.append({'round': tag, 'model': name, 'structure': structure,
                     'params': params, 'test_mae': test_mae, 'test_mse': test_mse,
                     'train_time_s': round(elapsed, 2), 'best_epoch': best_epoch,
                     'best_val_loss': round(best_val, 6), 'split_mode': 'time-first'})
        save_test_pred_vs_true(tag, name, pred_orig, true_orig)
        print(f'  params={params}  test MAE={test_mae:.4f}  test MSE={test_mse:.4f}  '
              f'best_epoch(based on val)={best_epoch}  time={elapsed:.1f}s')

    # 追加到 final_test_results.csv（每次正式评价一行）
    df_new = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, 'final_test_results.csv')
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print('\n===== 最终 Test 正式评价结果 =====')
    print(df_new.to_string(index=False))
    print(f'\n结果已写入 {csv_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='最终方案在 Test 集上的正式评价')
    parser.add_argument('--config', default=os.path.join(BASE_DIR, 'config.yaml'))
    parser.add_argument('--model', action='append', required=True,
                        help='最终选定的模型名，可多次指定，如 --model linear --model lstm')
    parser.add_argument('--round', default='final',
                        help='评价批次标识，默认 final')
    parser.add_argument('--epochs', type=int, default=None,
                        help='临时覆盖 config 的 epochs（快速验证用）')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
        print(f'[cli] --epochs 覆盖为 {args.epochs}（快速验证用）')

    run(config, args.round, args.model)
