"""Real ETTh1 training adapter for the declarative NAS engine.

All candidates share one chronological train/validation split and train-only
normalization. The untouched final test tail is never exposed to NAS ranking.
"""
from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path
from typing import Any

import numpy as np


def build_etth1_nas_runner(*, input_len: int = 96, pred_len: int = 24,
                           epochs: int = 5, batch_size: int = 128,
                           patience: int = 2, device: str | None = None):
    """Return (train_evaluate, split_hash) for run_nas_search on bundled ETTh1."""
    if min(input_len, pred_len, epochs, batch_size, patience) <= 0:
        raise ValueError("input_len, pred_len, epochs, batch_size and patience must be positive")

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from train_ett_corpus import load_series, build_group, TrainArgs
    from .architecture import build_model
    from .nas import build_optimizer, build_scheduler

    root = Path(__file__).resolve().parent.parent
    csv_path = root / "ETTh1.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Bundled ETTh1.csv not found: {csv_path}")
    series = load_series(csv_path, "ETTh1", expected_minutes=60)
    cfg = TrainArgs(input_len=input_len, pred_len=pred_len, epochs=epochs,
                    batch_size=batch_size, patience=patience)
    data = build_group([series], cfg)
    split_material = (f"{series['sha256']}|chronological_60_20_20|{input_len}|{pred_len}|"
                     f"{cfg.train_stride}|{cfg.eval_stride}|train_prefix_only_scaler")
    split_hash = hashlib.sha256(split_material.encode()).hexdigest()

    train_x = torch.from_numpy(data["train_x"])
    train_y = torch.from_numpy(data["train_y"])
    val_x_np, val_y_np = data["val"]["ETTh1"]
    val_x = torch.from_numpy(val_x_np)
    val_y = torch.from_numpy(val_y_np)
    # V9: reserve separate chronological validation windows for architecture
    # updates and checkpoint selection; the latter never updates architecture logits.
    arch_cut = max(1, len(val_x) // 2)
    arch_val_x, arch_val_y = val_x[:arch_cut], val_y[:arch_cut]
    select_val_x, select_val_y = val_x[arch_cut:], val_y[arch_cut:]
    if len(select_val_x) == 0:
        raise ValueError("validation split is too small for bilevel NAS")
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(target_device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {target_device}")

    def train_evaluate(spec: dict[str, Any], seed: int, loss_fn):
        run_started = time.perf_counter()
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size,
                            shuffle=True, generator=generator, drop_last=False)
        model = build_model(spec, input_len=input_len, n_features=7,
                            pred_len=pred_len).to(target_device)
        # V9: genuine first-order bilevel updates. Network weights are optimized
        # on train batches; DARTS architecture logits are optimized on a disjoint
        # architecture-validation stream. Checkpoint selection uses a third role
        # (the held-out half of the validation segment).
        arch_params = [p for n, p in model.named_parameters() if "architecture_logits" in n]
        arch_param_ids = {id(p) for p in arch_params}
        weight_params = [p for p in model.parameters() if id(p) not in arch_param_ids]
        optimizer = build_optimizer(type("WeightView", (), {"parameters": lambda self: iter(weight_params)})(), spec["config"])
        arch_optimizer = torch.optim.Adam(arch_params, lr=max(1e-5, float(spec["config"].get("learning_rate", 1e-3)) * 0.1)) if arch_params else None
        scheduler = build_scheduler(optimizer, spec["config"])
        best_state = None; best_val = float("inf"); best_metrics = None
        stale = 0; epochs_done = 0
        for epoch in range(epochs):
            model.train()
            for batch_idx, (xb, yb) in enumerate(loader):
                xb = xb.to(target_device); yb = yb.to(target_device)
                if arch_params:
                    for p in arch_params: p.requires_grad_(False)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite NAS training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(weight_params, max_norm=1.0)
                optimizer.step()
                if arch_optimizer is not None:
                    for p in arch_params: p.requires_grad_(True)
                    for p in weight_params: p.requires_grad_(False)
                    ax = (batch_idx * batch_size) % max(1, len(arch_val_x))
                    avx = arch_val_x[ax:ax+batch_size]
                    avy = arch_val_y[ax:ax+batch_size]
                    if len(avx) == 0:
                        avx, avy = arch_val_x[:batch_size], arch_val_y[:batch_size]
                    avx, avy = avx.to(target_device), avy.to(target_device)
                    arch_optimizer.zero_grad(set_to_none=True)
                    arch_loss = loss_fn(model(avx), avy)
                    if not torch.isfinite(arch_loss):
                        raise FloatingPointError("non-finite DARTS architecture loss")
                    arch_loss.backward()
                    arch_optimizer.step()
                    for p in weight_params: p.requires_grad_(True)
            model.eval()
            val_losses=[]; abs_sum=0.0; sq_sum=0.0; count=0
            with torch.no_grad():
                for start in range(0, len(select_val_x), batch_size):
                    xb=select_val_x[start:start+batch_size].to(target_device)
                    yb=select_val_y[start:start+batch_size].to(target_device)
                    pred=model(xb)
                    vloss=loss_fn(pred,yb)
                    val_losses.append(float(vloss.detach().cpu()))
                    err=(pred-yb).float()
                    abs_sum += float(err.abs().sum().cpu())
                    sq_sum += float((err**2).sum().cpu())
                    count += err.numel()
            val_loss=float(np.mean(val_losses))
            metrics={"mae":abs_sum/max(1,count),"rmse":float(np.sqrt(sq_sum/max(1,count)))}
            epochs_done=epoch+1
            if val_loss < best_val:
                best_val=val_loss; best_metrics=metrics
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                stale=0
            else:
                stale += 1
            if scheduler is not None:
                if spec["config"].get("scheduler") == "plateau": scheduler.step(val_loss)
                else: scheduler.step()
            if stale >= patience: break
        if best_state is None or best_metrics is None:
            raise RuntimeError("NAS candidate did not produce a valid validation checkpoint")
        model.load_state_dict(best_state)
        architecture_result = None
        if arch_params:
            from .nas import extract_darts_genotype
            architecture_result = extract_darts_genotype(model, spec)
        return {"metrics": best_metrics, "status": "completed",
                "epochs_trained": epochs_done,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "train_seconds": time.perf_counter() - run_started,
                "optimizer": spec["config"].get("optimizer", "adamw"),
                "learning_rate": spec["config"].get("learning_rate", 1e-3),
                "scheduler": spec["config"].get("scheduler", "none"),
                "validation_windows": int(len(val_x_np)),
                "architecture_validation_windows": int(len(arch_val_x)),
                "checkpoint_selection_windows": int(len(select_val_x)),
                "bilevel_architecture_updates": int(epochs_done * len(loader)) if arch_params else 0,
                "bilevel_mode": "first_order_train_weights_arch_validation_logits" if arch_params else "not_applicable",
                "discrete_architecture": architecture_result,
                "validation_protocol": "chronological_60_20_20_train_prefix_scaler_with_disjoint_arch_and_selection_halves"}

    return train_evaluate, split_hash


def run_etth1_nas(*, population_size: int = 8, generations: int = 2,
                  seeds: list[int] | None = None, epochs: int = 5,
                  batch_size: int = 128, patience: int = 2,
                  device: str | None = None, parents: list[dict[str, Any]] | None = None,
                  weakness: str = ""):
    """Execute real ETTh1 validation NAS; final test remains untouched."""
    from .architecture import run_nas_search
    runner, split_hash = build_etth1_nas_runner(epochs=epochs, batch_size=batch_size,
                                               patience=patience, device=device)
    return run_nas_search(train_evaluate=runner, ablation_evaluate=runner,
                          split_hash=split_hash, dataset="ETTh1",
                          population_size=population_size, generations=generations,
                          seeds=seeds or [11, 22, 33], parents=parents or None,
                          metric="mae", weakness=weakness)
