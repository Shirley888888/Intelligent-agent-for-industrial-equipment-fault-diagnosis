"""Bounded symbolic regression for interpretable time-series residual correction.

No eval/exec or generated source is used. Expressions come from a fixed grammar,
are fit on a chronological training prefix, and are selected on a holdout suffix.
"""
from __future__ import annotations
import numpy as np


def _basis(x: np.ndarray):
    n, f = x.shape
    t = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    cols = [np.ones(n), t, t*t, np.sin(2*np.pi*t), np.cos(2*np.pi*t)]
    names = ["1", "t", "t^2", "sin(2*pi*t)", "cos(2*pi*t)"]
    for j in range(f):
        v = x[:, j]
        cols.extend([v, v*v, np.abs(v)])
        names.extend([f"x{j}", f"x{j}^2", f"abs(x{j})"])
        if n > 1:
            d = np.diff(v, prepend=v[0])
            cols.append(d); names.append(f"diff(x{j})")
    for i in range(f):
        for j in range(i+1, f):
            cols.append(x[:, i] * x[:, j]); names.append(f"x{i}*x{j}")
    matrix = np.column_stack(cols)
    # Standardize non-constant columns for stable sparse selection.
    scale = np.std(matrix, axis=0)
    scale[scale < 1e-10] = 1.0
    return matrix / scale, names, scale


def search_symbolic_residual(inputs, residual, *, max_terms: int = 5, ridge: float = 1e-6):
    x = np.asarray(inputs, dtype=np.float64)
    y = np.asarray(residual, dtype=np.float64).reshape(-1)
    if x.ndim == 1: x = x[:, None]
    if x.ndim != 2 or len(x) != len(y) or len(y) < 20:
        raise ValueError("inputs must be [samples, features] and align with >=20 residuals")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("symbolic regression inputs must be finite")
    if not 1 <= int(max_terms) <= 12 or ridge < 0:
        raise ValueError("max_terms must be 1..12 and ridge must be non-negative")
    design, names, scale = _basis(x)
    split = max(10, min(len(y)-5, int(len(y)*0.8)))
    train_x, val_x = design[:split], design[split:]
    train_y, val_y = y[:split], y[split:]
    selected, remaining = [], list(range(1, design.shape[1]))
    # Greedy orthogonal matching pursuit on training prefix only.
    for _ in range(min(max_terms, len(remaining))):
        best = None
        for idx in remaining:
            trial = selected + [idx]
            coef = np.linalg.solve(train_x[:, trial].T @ train_x[:, trial] + ridge*np.eye(len(trial)),
                                   train_x[:, trial].T @ train_y)
            score = np.mean((train_y - train_x[:, trial] @ coef)**2)
            if best is None or score < best[0]: best = (score, idx)
        if best is None: break
        selected.append(best[1]); remaining.remove(best[1])
    if not selected: selected = [0]
    coef = np.linalg.solve(train_x[:, selected].T @ train_x[:, selected] + ridge*np.eye(len(selected)),
                           train_x[:, selected].T @ train_y)
    pred = val_x[:, selected] @ coef
    mae = float(np.mean(np.abs(val_y-pred)))
    expression = " + ".join(f"({float(c):+.6g})*{names[i]}" for c, i in zip(coef, selected))
    return {"expression": expression, "terms": [{"name": names[i], "coefficient": float(c)} for i, c in zip(selected, coef)],
            "validation_mae": mae, "train_samples": int(split), "validation_samples": int(len(y)-split),
            "selection": "greedy_forward_train_prefix_then_chronological_validation",
            "grammar": "constant/time/Fourier/raw/square/abs/difference/pairwise_product"}
