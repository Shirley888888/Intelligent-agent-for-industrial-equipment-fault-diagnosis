#!/usr/bin/env python3
"""Train shared LSTM/TCN oil-temperature forecasters on real ETT-small series.

Hourly (ETTh1+ETTh2) and 15-minute (ETTm1+ETTm2) corpora are trained separately.
All splits are chronological per transformer; windows never cross series boundaries.
The scaler is fitted only on each series' training prefix. Validation selects epochs;
the held-out test tail is evaluated once after training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from models import LSTMModel, TCNModel

COLS = ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
FEATURES = COLS[1:]
TARGET_INDEX = FEATURES.index("OT")
DATASETS = {"hourly": ["ETTh1", "ETTh2"], "15min": ["ETTm1", "ETTm2"]}
FREQ_MINUTES = {"hourly": 60, "15min": 15}
SOURCE_BASE = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"


@dataclass
class TrainArgs:
    model: str = "LSTM"
    input_len: int = 96
    pred_len: int = 24
    epochs: int = 10
    patience: int = 3
    batch_size: int = 128
    lr: float = 0.001
    weight_decay: float = 1e-5
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    tcn_channels: tuple[int, ...] = (64, 64, 128)
    kernel_size: int = 5
    train_stride: int = 1
    eval_stride: int = 2
    seed: int = 42

    def validate(self) -> None:
        if self.model not in {"LSTM", "TCN"}:
            raise ValueError("model must be LSTM or TCN")
        if min(self.input_len, self.pred_len, self.epochs, self.patience,
               self.batch_size, self.hidden_size, self.num_layers,
               self.train_stride, self.eval_stride) <= 0:
            raise ValueError("Length, epoch, batch, model width, patience and strides must be positive")
        if not 0 < self.lr or not 0 <= self.weight_decay or not 0 <= self.dropout < 1:
            raise ValueError("Invalid optimizer/dropout setting")
        if not self.tcn_channels or any(c <= 0 for c in self.tcn_channels):
            raise ValueError("tcn_channels must contain positive widths")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_series(path: Path, dataset_name: str, expected_minutes: int) -> dict[str, Any]:
    values: list[list[float]] = []
    timestamps: list[datetime] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLS:
            raise ValueError(f"{path}: expected columns {COLS}, got {reader.fieldnames}")
        for line, row in enumerate(reader, start=2):
            t = datetime.fromisoformat(row["date"])
            vector = [float(row[name]) for name in FEATURES]
            if not np.isfinite(vector).all():
                raise ValueError(f"{path}:{line}: non-finite numeric value")
            if timestamps:
                delta = (t - timestamps[-1]).total_seconds() / 60
                if delta != expected_minutes:
                    raise ValueError(f"{path}:{line}: expected {expected_minutes}-minute cadence, got {delta}")
            timestamps.append(t)
            values.append(vector)
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) < 1000:
        raise ValueError(f"{path}: too few records ({len(arr)})")
    return {"name": dataset_name, "path": path, "values": arr, "timestamps": timestamps,
            "sha256": sha256(path), "rows": len(arr),
            "start": timestamps[0].isoformat(), "end": timestamps[-1].isoformat()}


def split_indices(n: int) -> tuple[int, int]:
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    return train_end, val_end


def make_windows(values: np.ndarray, start_origin: int, end: int,
                 input_len: int, pred_len: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    origins = np.arange(max(input_len, start_origin), end - pred_len + 1, stride, dtype=np.int64)
    if len(origins) == 0:
        raise ValueError("Not enough rows in split to construct complete windows")
    x = np.stack([values[o-input_len:o] for o in origins]).astype(np.float32)
    y = np.stack([values[o:o+pred_len, TARGET_INDEX] for o in origins]).astype(np.float32)
    return x, y


def build_group(series_list: list[dict[str, Any]], cfg: TrainArgs):
    # Independent chronological split inside each physical transformer series.
    ranges = {}
    train_rows = []
    for s in series_list:
        n = len(s["values"])
        train_end, val_end = split_indices(n)
        if train_end < cfg.input_len + cfg.pred_len + 1 or val_end-train_end < cfg.pred_len:
            raise ValueError(f"{s['name']}: insufficient data for chronological train/val/test split")
        ranges[s["name"]] = {"train_end": train_end, "val_end": val_end, "test_end": n}
        train_rows.append(s["values"][:train_end])
    fit_rows = np.concatenate(train_rows, axis=0)
    mean = fit_rows.mean(axis=0).astype(np.float32)
    scale = fit_rows.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0

    data = {"train_x": [], "train_y": [], "val": {}, "test": {}}
    for s in series_list:
        v = s["values"]
        r = ranges[s["name"]]
        z = (v - mean) / scale
        # Training targets and histories stay fully inside train prefix.
        tx, ty = make_windows(z, cfg.input_len, r["train_end"], cfg.input_len,
                              cfg.pred_len, cfg.train_stride)
        # Holdout targets start at the split boundary; history may use past rows only.
        vx, vy = make_windows(z, r["train_end"], r["val_end"], cfg.input_len,
                              cfg.pred_len, cfg.eval_stride)
        ex, ey = make_windows(z, r["val_end"], r["test_end"], cfg.input_len,
                              cfg.pred_len, cfg.eval_stride)
        data["train_x"].append(tx); data["train_y"].append(ty)
        data["val"][s["name"]] = (vx, vy)
        data["test"][s["name"]] = (ex, ey)
    data["train_x"] = np.concatenate(data["train_x"], axis=0)
    data["train_y"] = np.concatenate(data["train_y"], axis=0)
    data["mean"], data["scale"], data["ranges"] = mean, scale, ranges
    return data


def make_model(cfg: TrainArgs):
    if cfg.model == "LSTM":
        return LSTMModel(cfg.input_len, len(FEATURES), cfg.pred_len,
                         hidden_size=cfg.hidden_size, num_layers=cfg.num_layers,
                         dropout=cfg.dropout)
    return TCNModel(cfg.input_len, len(FEATURES), cfg.pred_len,
                    channels=tuple(cfg.tcn_channels), kernel_size=cfg.kernel_size,
                    dropout=cfg.dropout)


def regression_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    err = np.abs(pred.reshape(-1) - truth.reshape(-1))
    signed = pred.reshape(-1) - truth.reshape(-1)
    return {"MAE": float(np.mean(err)), "RMSE": float(np.sqrt(np.mean(signed ** 2))),
            "P95_AE": float(np.quantile(err, 0.95)), "P99_AE": float(np.quantile(err, 0.99)),
            "MaxAE": float(np.max(err))}


def train_one(series_list: list[dict[str, Any]], frequency: str, cfg: TrainArgs,
              out_dir: Path, device: str) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    cfg.validate()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(cfg.seed)
    torch.use_deterministic_algorithms(True)
    corpus = build_group(series_list, cfg)
    mean, scale = corpus["mean"], corpus["scale"]
    # build_group already normalizes every split with train-prefix-only statistics.
    train_x = corpus["train_x"]
    train_y = corpus["train_y"]
    net = make_model(cfg).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
                        batch_size=cfg.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(cfg.seed))
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = float("inf"); best_state = None; best_epoch = 0; stale = 0
    epoch_log = []
    for epoch in range(1, cfg.epochs + 1):
        net.train(); batch_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(net(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); batch_losses.append(float(loss.detach().cpu()))
        net.eval(); v_errors = []
        with torch.no_grad():
            for _, (vx, vy) in corpus["val"].items():
                pred_z = []
                for i in range(0, len(vx), 512):
                    pred_z.append(net(torch.from_numpy(vx[i:i+512]).to(device)).cpu().numpy())
                p = np.concatenate(pred_z) * scale[TARGET_INDEX] + mean[TARGET_INDEX]
                y = vy * scale[TARGET_INDEX] + mean[TARGET_INDEX]
                v_errors.extend(np.abs(p.reshape(-1)-y.reshape(-1)).tolist())
        val_mae = float(np.mean(v_errors))
        epoch_log.append({"epoch": epoch, "train_mse_normalized": float(np.mean(batch_losses)),
                          "validation_MAE_degC": val_mae})
        print(f"[{frequency}/{cfg.model}] epoch {epoch}/{cfg.epochs} train_mse={np.mean(batch_losses):.6f} val_MAE={val_mae:.5f}", flush=True)
        if val_mae < best_val - 1e-8:
            best_val, best_epoch, stale = val_mae, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a finite best checkpoint")
    net.load_state_dict(best_state); net.eval()

    per_series = {}
    aggregate_pred, aggregate_true = [], []
    with torch.no_grad():
        for s in series_list:
            vx, vy = corpus["test"][s["name"]]
            pz = []
            for i in range(0, len(vx), 512):
                pz.append(net(torch.from_numpy(vx[i:i+512]).to(device)).cpu().numpy())
            pred = np.concatenate(pz) * scale[TARGET_INDEX] + mean[TARGET_INDEX]
            truth = vy * scale[TARGET_INDEX] + mean[TARGET_INDEX]
            per_series[s["name"]] = {**regression_metrics(pred, truth), "test_windows": int(len(truth)),
                                      "rows": int(s["rows"]), "source_sha256": s["sha256"]}
            aggregate_pred.append(pred); aggregate_true.append(truth)
    all_pred = np.concatenate(aggregate_pred); all_true = np.concatenate(aggregate_true)
    test_metrics = {"aggregate": {**regression_metrics(all_pred, all_true),
                                  "series_count": len(series_list),
                                  "test_windows": int(sum(x["test_windows"] for x in per_series.values()))},
                    "per_series": per_series}

    out_dir.mkdir(parents=True, exist_ok=False)
    interval = FREQ_MINUTES[frequency]
    meta = {"schema_version": 1, "task": "transformer_oil_temperature_forecasting",
            "model_name": cfg.model, "features": FEATURES, "target": "OT", "target_unit": "degC",
            "input_len": cfg.input_len, "pred_len": cfg.pred_len,
            "sample_interval_minutes": interval,
            "input_duration_hours": cfg.input_len * interval / 60,
            "forecast_duration_hours": cfg.pred_len * interval / 60,
            "train_config": asdict(cfg), "normalization": "global_train_prefix_only",
            "data_sources": [{"dataset": s["name"], "source_url": f"{SOURCE_BASE}/{s['name']}.csv",
                              "source_sha256": s["sha256"], "rows": s["rows"],
                              "start": s["start"], "end": s["end"]} for s in series_list],
            "split_policy": "chronological_per_series_60_20_20",
            "split_ranges": corpus["ranges"]}
    checkpoint = out_dir / "checkpoint.pt"
    torch.save({"model_state": best_state, "mean": mean.tolist(), "scale": scale.tolist(),
                "metadata": meta}, checkpoint)
    manifest = {"model_id": f"ETT-{frequency}-{cfg.model}-v1", "created_at": datetime.now().isoformat(timespec="seconds"),
                "checkpoint": checkpoint.name, "checkpoint_sha256": sha256(checkpoint),
                "frequency": frequency, "model_name": cfg.model, "target": "OT",
                "target_unit": "degC", "sample_interval_minutes": interval,
                "best_epoch": best_epoch, "best_validation_MAE_degC": best_val,
                "test_metrics": test_metrics, "data_sources": meta["data_sources"],
                "split_policy": meta["split_policy"], "normalization": meta["normalization"],
                "license_note": "Source dataset is CC BY-ND 4.0; see ETT_DATA_LICENSE.md. Verify redistribution rights for model artifacts before publishing."}
    (out_dir / "training_history.json").write_text(json.dumps(epoch_log, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SAVED] {checkpoint} sha256={manifest['checkpoint_sha256']}")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Train LSTM/TCN on the official ETT-small oil-temperature data")
    p.add_argument("--data-dir", default="data/ett")
    p.add_argument("--frequency", choices=["hourly", "15min", "all"], default="hourly")
    p.add_argument("--model", choices=["LSTM", "TCN", "both"], default="both")
    p.add_argument("--datasets", nargs="+", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
                   help="Optional explicit subset, e.g. ETTh1 for local smoke training")
    p.add_argument("--out-dir", default="models/ett_corpus")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--train-stride", type=int, default=1)
    p.add_argument("--eval-stride", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    data_dir, out_root = Path(args.data_dir), Path(args.out_dir)
    frequencies = ["hourly", "15min"] if args.frequency == "all" else [args.frequency]
    models = ["LSTM", "TCN"] if args.model == "both" else [args.model]
    results = []
    for freq in frequencies:
        names = args.datasets or DATASETS[freq]
        expected = "hourly" if freq == "hourly" else "15min"
        if any(FILES_FREQ(n) != expected for n in names):
            raise ValueError(f"Datasets {names} do not match frequency group {freq}")
        if args.datasets and freq == "all":
            raise ValueError("Do not combine explicit --datasets with --frequency all; use one frequency per run")
        series = []
        for name in names:
            path = data_dir / f"{name}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}. Run: python download_ett.py --output-dir {data_dir}")
            series.append(load_series(path, name, FREQ_MINUTES[freq]))
        for model_name in models:
            stride = args.train_stride
            if freq == "15min" and args.train_stride == 1:
                stride = 4  # control memory and reduce redundant 15-minute windows
            cfg = TrainArgs(model=model_name, epochs=args.epochs, patience=args.patience,
                            batch_size=args.batch_size, hidden_size=args.hidden_size,
                            num_layers=args.num_layers, dropout=args.dropout,
                            train_stride=stride, eval_stride=args.eval_stride, seed=args.seed)
            out = out_root / freq / model_name
            results.append(train_one(series, freq, cfg, out, args.device))
    print(json.dumps({"runs": results}, ensure_ascii=False, indent=2))


def FILES_FREQ(name: str) -> str:
    return "hourly" if name.startswith("ETTh") else "15min"


if __name__ == "__main__":
    main()
