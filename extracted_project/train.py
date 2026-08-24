import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import download_etth1, SlidingWindowDataset
from models import LinearModel, MLP, Conv1DModel, LSTMModel


def collate_fn(batch):
    xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]
    xs = torch.tensor(np.stack(xs))
    ys = torch.tensor(np.stack(ys))
    return xs, ys


def train_one_epoch(model, loader, optim, loss_fn, device):
    model.train()
    total_loss = 0.0
    for x,y in loader:
        x = x.to(device); y = y.to(device)
        optim.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optim.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def eval_model(model, loader, device):
    model.eval()
    preds=[]; trues=[]
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            p = model(x).cpu().numpy()
            preds.append(p)
            trues.append(y.numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    # compute MAE/MSE across flattened predictions
    mae = mean_absolute_error(trues.flatten(), preds.flatten())
    mse = mean_squared_error(trues.flatten(), preds.flatten())
    return mae, mse, preds, trues


def run(args):
    data_path = download_etth1(dest=os.path.join('data','ETTh1.csv'))
    ds = SlidingWindowDataset(data_path, input_len=args.input_len, pred_len=args.pred_len)
    train_idx, val_idx, test_idx = ds.split_timebased()

    def subset_loader(idxs, batch_size, shuffle=False):
        subset = torch.utils.data.Subset(ds, idxs)
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    train_loader = subset_loader(train_idx, args.batch_size, shuffle=True)
    val_loader = subset_loader(val_idx, args.batch_size, shuffle=False)
    test_loader = subset_loader(test_idx, args.batch_size, shuffle=False)

    device = torch.device(args.device if torch.cuda.is_available() and args.device!='cpu' else 'cpu')

    # create model
    n_features = len(ds.features)
    if args.model=='baseline':
        model = None
    elif args.model=='linear':
        model = LinearModel(input_len=args.input_len, n_features=n_features, pred_len=args.pred_len)
    elif args.model=='mlp':
        model = MLP(input_len=args.input_len, n_features=n_features, hidden=args.hidden, n_hidden_layers=args.layers, pred_len=args.pred_len)
    elif args.model=='cnn':
        model = Conv1DModel(in_channels=n_features, out_channels=args.channels, kernel_size=args.kernel, pred_len=args.pred_len)
    elif args.model=='lstm':
        model = LSTMModel(n_features=n_features, hidden_size=args.hidden, num_layers=args.layers, pred_len=args.pred_len)
    else:
        raise ValueError('Unknown model')

    if model is not None:
        model = model.to(device)
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.MSELoss()

    os.makedirs('results', exist_ok=True)

    # baseline (last value persistence)
    def baseline_eval(loader):
        preds=[]; trues=[]
        for x,y in loader:
            # last time step value from input, repeat pred_len times
            last = x[:, -1, -1].numpy()  # take last feature's last value as proxy
            p = np.repeat(last[:,None], args.pred_len, axis=1)
            preds.append(p); trues.append(y.numpy())
        preds=np.concatenate(preds,axis=0); trues=np.concatenate(trues,axis=0)
        mae = mean_absolute_error(trues.flatten(), preds.flatten())
        mse = mean_squared_error(trues.flatten(), preds.flatten())
        return mae, mse, preds, trues

    results_rows = []

    if args.model=='baseline':
        mae, mse, preds, trues = baseline_eval(test_loader)
        results_rows.append({'model':'baseline','mae':mae,'mse':mse,'params':0,'time':0.0})
    else:
        best_val = float('inf'); best_state=None; start_time=time.time()
        for epoch in range(1, args.epochs+1):
            tr_loss = train_one_epoch(model, train_loader, optim, loss_fn, device)
            val_mae, val_mse, _, _ = eval_model(model, val_loader, device)
            print(f"Epoch {epoch}/{args.epochs}  train_loss={tr_loss:.4f}  val_mae={val_mae:.4f}")
            # simple early save
            if val_mae < best_val:
                best_val = val_mae
                best_state = model.state_dict()
        elapsed = time.time()-start_time
        # load best
        if best_state is not None:
            model.load_state_dict(best_state)
        test_mae, test_mse, preds, trues = eval_model(model, test_loader, device)
        # count params
        params = sum(p.numel() for p in model.parameters())
        results_rows.append({'model':args.model,'mae':test_mae,'mse':test_mse,'params':params,'time':elapsed})

    # save metrics and a small plot for the first test sample
    df = pd.DataFrame(results_rows)
    df.to_csv('results/metrics.csv', index=False)

    # plot first test example predictions vs truth (flatten first sample)
    try:
        os.makedirs('results/figures', exist_ok=True)
        # pick sample 0
        p0 = preds[0]
        t0 = trues[0]
        plt.figure(figsize=(8,3))
        plt.plot(range(len(t0)), t0, label='truth')
        plt.plot(range(len(p0)), p0, label='pred')
        plt.legend()
        plt.title(f'{args.model} first test sample')
        plt.savefig('results/figures/pred_vs_true.png', bbox_inches='tight')
        plt.close()
    except Exception as e:
        print('Failed to save figure:', e)

    print('Results written to results/metrics.csv and results/figures/pred_vs_true.png (if available)')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mlp', choices=['baseline','linear','mlp','cnn','lstm'])
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--kernel', type=int, default=5)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    run(args)
