"""Trainable, versioned multi-task LSTM/TCN models for industrial fault diagnosis.

The trainer requires externally prepared, time/group-disjoint labelled windows.
It does not manufacture industrial labels from ETTh1 forecasting targets.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass
class IndustrialFaultTrainConfig:
    model_name: str
    fault_classes: list[str]
    n_features: int
    input_len: int = 96
    hidden_size: int = 96
    num_layers: int = 2
    tcn_channels: tuple[int, ...] = (32, 64, 64)
    kernel_size: int = 3
    dropout: float = 0.1
    epochs: int = 20
    patience: int = 5
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    seed: int = 42
    device: str = "cpu"
    split_policy: str = "chronological_disjoint"
    rul_unit: str = "hour"
    amplitude_unit: str = "dataset_native"
    calibration_min_samples: int = 100
    calibration_max_ece: float = 0.15
    require_confidence_calibration: bool = True
    min_val_accuracy: float = 0.70
    min_class_recall: float = 0.50
    min_class_support: int = 5
    max_rul_mae: float | None = None
    max_amplitude_mae: float | None = None
    min_anomaly_auroc: float | None = None

    def validate(self) -> None:
        _require(self.model_name in {"LSTM", "TCN"}, "model_name must be LSTM or TCN")
        _require(isinstance(self.fault_classes, list) and len(self.fault_classes) >= 2,
                 "fault_classes must contain at least two class names")
        _require(all(isinstance(x, str) and x.strip() for x in self.fault_classes),
                 "fault class names must be nonempty strings")
        _require(len(set(self.fault_classes)) == len(self.fault_classes), "fault_classes must be unique")
        for name in ("n_features", "input_len", "hidden_size", "num_layers", "epochs", "patience", "batch_size"):
            _require(type(getattr(self, name)) is int and getattr(self, name) > 0, f"{name} must be positive")
        _require(isinstance(self.tcn_channels, (list, tuple)) and len(self.tcn_channels) > 0
                 and all(type(x) is int and x > 0 for x in self.tcn_channels),
                 "tcn_channels must contain positive integers")
        _require(type(self.kernel_size) is int and self.kernel_size > 0, "kernel_size must be positive")
        _require(0 <= self.dropout < 1, "dropout must be in [0,1)")
        _require(np.isfinite(self.learning_rate) and self.learning_rate > 0, "learning_rate must be positive")
        _require(np.isfinite(self.weight_decay) and self.weight_decay >= 0, "weight_decay must be nonnegative")
        _require(self.split_policy in {"chronological_disjoint", "equipment_group_disjoint"},
                 "split_policy must explicitly assert chronological_disjoint or equipment_group_disjoint")
        _require(bool(self.rul_unit.strip()) and bool(self.amplitude_unit.strip()), "target units are required")
        _require(type(self.calibration_min_samples) is int and self.calibration_min_samples >= 20,
                 "calibration_min_samples must be >= 20")
        _require(0 < self.calibration_max_ece < 1, "calibration_max_ece must be in (0,1)")
        _require(type(self.require_confidence_calibration) is bool, "require_confidence_calibration must be bool")
        _require(0 <= self.min_val_accuracy <= 1, "min_val_accuracy must be in [0,1]")
        _require(0 <= self.min_class_recall <= 1, "min_class_recall must be in [0,1]")
        _require(type(self.min_class_support) is int and self.min_class_support >= 1,
                 "min_class_support must be >= 1")
        for name in ("max_rul_mae", "max_amplitude_mae", "min_anomaly_auroc"):
            value = getattr(self, name)
            if value is not None:
                _require(np.isfinite(value) and value >= 0, f"{name} must be finite and nonnegative")
        if self.min_anomaly_auroc is not None:
            _require(self.min_anomaly_auroc <= 1, "min_anomaly_auroc must be <= 1")


def _torch_classes():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("PyTorch is required for industrial fault model training/inference") from exc

    class FaultLSTM(nn.Module):
        def __init__(self, n_features: int, n_classes: int, hidden_size: int,
                     num_layers: int, dropout: float):
            super().__init__()
            self.encoder = nn.LSTM(n_features, hidden_size, num_layers=num_layers,
                                   batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
            self.dropout = nn.Dropout(dropout)
            self.fault_head = nn.Linear(hidden_size, n_classes)
            self.anomaly_head = nn.Linear(hidden_size, 1)
            self.rul_head = nn.Linear(hidden_size, 1)
            self.amplitude_head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            encoded, _ = self.encoder(x)
            h = self.dropout(encoded[:, -1, :])
            return {"fault_logits": self.fault_head(h),
                    "anomaly_logits": self.anomaly_head(h).squeeze(-1),
                    "rul_scaled": torch.nn.functional.softplus(self.rul_head(h).squeeze(-1)),
                    "amplitude_scaled": torch.nn.functional.softplus(self.amplitude_head(h).squeeze(-1))}

    class FaultTCN(nn.Module):
        def __init__(self, n_features: int, n_classes: int, channels: tuple[int, ...],
                     kernel_size: int, dropout: float):
            super().__init__()
            from models import TemporalBlock
            blocks = []
            in_ch = n_features
            for index, out_ch in enumerate(channels):
                blocks.append(TemporalBlock(in_ch, int(out_ch), kernel_size, 2 ** index, dropout))
                in_ch = int(out_ch)
            self.encoder = nn.Sequential(*blocks)
            self.dropout = nn.Dropout(dropout)
            self.fault_head = nn.Linear(in_ch, n_classes)
            self.anomaly_head = nn.Linear(in_ch, 1)
            self.rul_head = nn.Linear(in_ch, 1)
            self.amplitude_head = nn.Linear(in_ch, 1)

        def forward(self, x):
            h = self.dropout(self.encoder(x.transpose(1, 2))[:, :, -1])
            return {"fault_logits": self.fault_head(h),
                    "anomaly_logits": self.anomaly_head(h).squeeze(-1),
                    "rul_scaled": torch.nn.functional.softplus(self.rul_head(h).squeeze(-1)),
                    "amplitude_scaled": torch.nn.functional.softplus(self.amplitude_head(h).squeeze(-1))}

    return torch, FaultLSTM, FaultTCN


def build_fault_network(config: IndustrialFaultTrainConfig):
    torch, FaultLSTM, FaultTCN = _torch_classes()
    if config.model_name == "LSTM":
        return FaultLSTM(config.n_features, len(config.fault_classes), config.hidden_size,
                         config.num_layers, config.dropout)
    return FaultTCN(config.n_features, len(config.fault_classes), tuple(config.tcn_channels),
                    config.kernel_size, config.dropout)


def _load_dataset(dataset_path: str | Path, cfg: IndustrialFaultTrainConfig) -> dict[str, np.ndarray | None]:
    with np.load(dataset_path, allow_pickle=False) as archive:
        required = ("X_train", "X_val", "y_fault_train", "y_fault_val")
        missing = [key for key in required if key not in archive]
        _require(not missing, f"Dataset missing arrays: {missing}")
        result: dict[str, np.ndarray | None] = {}
        for key in required:
            result[key] = np.asarray(archive[key])
        for target in ("y_anomaly", "y_rul_hours", "y_fault_amplitude"):
            for split in ("train", "val"):
                key = f"{target}_{split}"
                result[key] = np.asarray(archive[key]) if key in archive else None
        for split in ("cal_fit", "cal_eval"):
            for target in ("X", "y_fault"):
                key = f"{target}_{split}"
                result[key] = np.asarray(archive[key]) if key in archive else None
    for split in ("train", "val"):
        x = np.asarray(result[f"X_{split}"], dtype=np.float32)
        y = np.asarray(result[f"y_fault_{split}"])
        _require(x.ndim == 3 and x.shape[1:] == (cfg.input_len, cfg.n_features),
                 f"X_{split} must have shape [N,{cfg.input_len},{cfg.n_features}]")
        _require(len(x) >= 1 and np.isfinite(x).all(), f"X_{split} must be nonempty and finite")
        _require(y.shape == (len(x),) and np.issubdtype(y.dtype, np.integer),
                 f"y_fault_{split} must be an integer vector aligned with X_{split}")
        _require(np.all((y >= 0) & (y < len(cfg.fault_classes))),
                 f"y_fault_{split} values must index fault_classes")
        result[f"X_{split}"] = x
        result[f"y_fault_{split}"] = y.astype(np.int64)
        for target in ("y_anomaly", "y_rul_hours", "y_fault_amplitude"):
            values = result[f"{target}_{split}"]
            if values is not None:
                values = np.asarray(values, dtype=np.float32)
                _require(values.shape == (len(x),), f"{target}_{split} must be a vector aligned with X_{split}")
                finite = values[np.isfinite(values)]
                if target == "y_anomaly":
                    _require(np.all((finite == 0) | (finite == 1)), f"{target}_{split} must contain only 0/1/NaN")
                else:
                    _require(np.all(finite >= 0), f"{target}_{split} must be nonnegative/NaN")
                result[f"{target}_{split}"] = values
    for split in ("cal_fit", "cal_eval"):
        x = result[f"X_{split}"]
        y = result[f"y_fault_{split}"]
        _require((x is None) == (y is None), f"X_{split} and y_fault_{split} must be supplied together")
        if x is not None:
            x = np.asarray(x, dtype=np.float32)
            y = np.asarray(y)
            _require(x.ndim == 3 and x.shape[1:] == (cfg.input_len, cfg.n_features),
                     f"X_{split} must have shape [N,{cfg.input_len},{cfg.n_features}]")
            _require(len(x) >= 1 and np.isfinite(x).all(), f"X_{split} must be nonempty and finite")
            _require(y.shape == (len(x),) and np.issubdtype(y.dtype, np.integer),
                     f"y_fault_{split} must be an integer vector aligned with X_{split}")
            _require(np.all((y >= 0) & (y < len(cfg.fault_classes))),
                     f"y_fault_{split} values must index fault_classes")
            result[f"X_{split}"] = x
            result[f"y_fault_{split}"] = y.astype(np.int64)
    _require(len(np.unique(result["y_fault_train"])) >= 2, "training data must contain at least two fault classes")
    _require(len(np.unique(result["y_fault_val"])) >= 2, "validation data must contain at least two fault classes")
    return result


def _masked_loss(torch, pred, target, kind: str):
    mask = torch.isfinite(target)
    if not bool(mask.any()):
        return None
    if kind == "anomaly":
        return torch.nn.functional.binary_cross_entropy_with_logits(pred[mask], target[mask])
    return torch.nn.functional.smooth_l1_loss(pred[mask], target[mask])


def _expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if np.any(mask):
            accuracy = float(np.mean(prediction[mask] == labels[mask]))
            avg_conf = float(np.mean(confidence[mask]))
            ece += float(np.mean(mask)) * abs(accuracy - avg_conf)
    return float(ece)


def _binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = len(labels) - n_pos
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def train_industrial_fault_model(dataset_path: str | Path, config: IndustrialFaultTrainConfig,
                                 output_root: str | Path) -> dict[str, Any]:
    """Train a labelled industrial model and create immutable version M0.

    Dataset split must be prepared outside this function by time or equipment group.
    Training data and validation data are supplied as separate NPZ arrays.
    """
    config.validate()
    data = _load_dataset(dataset_path, config)
    torch, _, _ = _torch_classes()
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    x_train = np.asarray(data["X_train"], dtype=np.float32)
    x_val = np.asarray(data["X_val"], dtype=np.float32)
    mean = x_train.reshape(-1, config.n_features).mean(axis=0)
    scale = x_train.reshape(-1, config.n_features).std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    xtr = (x_train - mean) / scale
    xva = (x_val - mean) / scale

    rul_train = data["y_rul_hours_train"]
    amp_train = data["y_fault_amplitude_train"]
    rul_scale = max(float(np.nanmedian(rul_train)) if rul_train is not None and np.isfinite(rul_train).any() else 1.0, 1e-6)
    amp_scale = max(float(np.nanmedian(amp_train)) if amp_train is not None and np.isfinite(amp_train).any() else 1.0, 1e-6)

    device = torch.device(config.device)
    net = build_fault_network(config).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    tx = torch.tensor(xtr, dtype=torch.float32)
    ty = torch.tensor(data["y_fault_train"], dtype=torch.long)
    ds = torch.utils.data.TensorDataset(tx, ty)
    loader = torch.utils.data.DataLoader(ds, batch_size=config.batch_size,
                                         shuffle=True, num_workers=0)
    y_anomaly = data["y_anomaly_train"]
    y_rul = data["y_rul_hours_train"]
    y_amp = data["y_fault_amplitude_train"]
    val_fault = torch.tensor(data["y_fault_val"], dtype=torch.long, device=device)
    xval_tensor = torch.tensor(xva, dtype=torch.float32, device=device)
    best_loss, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0
    history: list[dict[str, float]] = []

    # Optional side-task targets are aligned to the same shuffled minibatch order.
    for epoch in range(1, config.epochs + 1):
        net.train()
        epoch_losses = []
        indices = torch.randperm(len(tx))
        for start in range(0, len(tx), config.batch_size):
            idx = indices[start:start + config.batch_size]
            xb = tx[idx].to(device)
            yb = ty[idx].to(device)
            output = net(xb)
            loss = torch.nn.functional.cross_entropy(output["fault_logits"], yb)
            if y_anomaly is not None:
                target = torch.tensor(np.asarray(y_anomaly)[idx.cpu().numpy()], dtype=torch.float32, device=device)
                part = _masked_loss(torch, output["anomaly_logits"], target, "anomaly")
                if part is not None:
                    loss = loss + 0.25 * part
            if y_rul is not None:
                raw = np.asarray(y_rul)[idx.cpu().numpy()]
                target = torch.tensor(raw / rul_scale, dtype=torch.float32, device=device)
                part = _masked_loss(torch, output["rul_scaled"], target, "regression")
                if part is not None:
                    loss = loss + 0.20 * part
            if y_amp is not None:
                raw = np.asarray(y_amp)[idx.cpu().numpy()]
                target = torch.tensor(raw / amp_scale, dtype=torch.float32, device=device)
                part = _masked_loss(torch, output["amplitude_scaled"], target, "regression")
                if part is not None:
                    loss = loss + 0.20 * part
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        net.eval()
        with torch.no_grad():
            val_out = net(xval_tensor)
            val_loss = float(torch.nn.functional.cross_entropy(val_out["fault_logits"], val_fault).cpu())
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(epoch_losses)), "val_ce": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break
    _require(best_state is not None, "No best checkpoint was produced")
    net.load_state_dict(best_state)
    net.eval()

    with torch.no_grad():
        val_outputs = net(xval_tensor)
        val_logits = val_outputs["fault_logits"].detach()
    val_labels_np = np.asarray(data["y_fault_val"], dtype=np.int64)
    temperature = 1.0
    calibration_error = None
    calibration_samples = 0
    calibration_id = None
    confidence_calibrated = False
    # Temperature fitting and ECE assessment use independent calibration arrays,
    # separate from train and release-validation data.
    x_cal_fit = data.get("X_cal_fit")
    y_cal_fit = data.get("y_fault_cal_fit")
    x_cal_eval = data.get("X_cal_eval")
    y_cal_eval = data.get("y_fault_cal_eval")
    if x_cal_fit is not None and x_cal_eval is not None:
        cal_fit_tensor = torch.tensor((x_cal_fit - mean) / scale, dtype=torch.float32, device=device)
        cal_eval_tensor = torch.tensor((x_cal_eval - mean) / scale, dtype=torch.float32, device=device)
        cal_fit_labels = torch.tensor(y_cal_fit, dtype=torch.long, device=device)
        with torch.no_grad():
            fit_logits = net(cal_fit_tensor)["fault_logits"].detach()
            eval_logits = net(cal_eval_tensor)["fault_logits"].detach()
        calibration_samples = len(y_cal_eval)
        if len(y_cal_fit) >= 20 and len(np.unique(y_cal_fit)) >= 2:
            log_temp = torch.zeros(1, device=device, requires_grad=True)
            optimizer_temp = torch.optim.LBFGS([log_temp], lr=0.05, max_iter=50)
            def closure():
                optimizer_temp.zero_grad()
                loss = torch.nn.functional.cross_entropy(
                    fit_logits / log_temp.exp().clamp(0.05, 20.0), cal_fit_labels)
                loss.backward()
                return loss
            try:
                optimizer_temp.step(closure)
                temperature = float(log_temp.detach().exp().clamp(0.05, 20.0).cpu())
            except Exception:
                temperature = 1.0
        eval_probs = torch.softmax(eval_logits / temperature, dim=1).cpu().numpy()
        eval_labels_np = np.asarray(y_cal_eval, dtype=np.int64)
        calibration_error = _expected_calibration_error(eval_probs, eval_labels_np)
        confidence_calibrated = bool(
            calibration_samples >= config.calibration_min_samples
            and len(np.unique(y_cal_eval)) >= 2
            and calibration_error <= config.calibration_max_ece
            and len(y_cal_fit) >= 20 and len(np.unique(y_cal_fit)) >= 2
        )
        calibration_hash = hashlib.sha256(
            np.asarray(x_cal_fit, dtype="<f4").tobytes()
            + np.asarray(y_cal_fit, dtype="<i8").tobytes()
            + np.asarray(x_cal_eval, dtype="<f4").tobytes()
            + np.asarray(y_cal_eval, dtype="<i8").tobytes()
        ).hexdigest()
        calibration_id = f"fault-temp-v1-{calibration_hash[:16]}" if confidence_calibrated else None
    # Validation accuracy/recall are evaluated without fitting the temperature there.
    val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()
    pred = val_probs.argmax(axis=1)
    per_class = {}
    for idx, name in enumerate(config.fault_classes):
        mask = val_labels_np == idx
        per_class[name] = {
            "support": int(mask.sum()),
            "recall": float(np.mean(pred[mask] == idx)) if np.any(mask) else None,
        }
    accuracy = float(np.mean(pred == val_labels_np))
    val_metrics: dict[str, Any] = {
        "accuracy": accuracy,
        "per_class": per_class,
        "cross_entropy": float(best_loss),
        "best_epoch": int(best_epoch),
        "calibration_ece": calibration_error,
    }
    anomaly_auc = None
    y_anomaly_val = data.get("y_anomaly_val")
    if y_anomaly_val is not None and np.isfinite(y_anomaly_val).any():
        anomaly_mask = np.isfinite(y_anomaly_val)
        anomaly_probs = torch.sigmoid(val_outputs["anomaly_logits"]).detach().cpu().numpy()
        anomaly_auc = _binary_auroc(np.asarray(y_anomaly_val)[anomaly_mask], anomaly_probs[anomaly_mask])
        val_metrics["anomaly_auroc"] = anomaly_auc
    rul_mae = None
    y_rul_val = data.get("y_rul_hours_val")
    if y_rul_val is not None and np.isfinite(y_rul_val).any():
        rul_mask = np.isfinite(y_rul_val)
        pred_rul = val_outputs["rul_scaled"].detach().cpu().numpy() * rul_scale
        rul_mae = float(np.mean(np.abs(pred_rul[rul_mask] - np.asarray(y_rul_val)[rul_mask])))
        val_metrics["rul_mae"] = rul_mae
    amplitude_mae = None
    y_amplitude_val = data.get("y_fault_amplitude_val")
    if y_amplitude_val is not None and np.isfinite(y_amplitude_val).any():
        amplitude_mask = np.isfinite(y_amplitude_val)
        pred_amplitude = val_outputs["amplitude_scaled"].detach().cpu().numpy() * amp_scale
        amplitude_mae = float(np.mean(np.abs(pred_amplitude[amplitude_mask] - np.asarray(y_amplitude_val)[amplitude_mask])))
        val_metrics["amplitude_mae"] = amplitude_mae
    class_gate = all(
        item["support"] >= config.min_class_support and item["recall"] is not None
        and item["recall"] >= config.min_class_recall
        for item in per_class.values()
    )
    calibration_gate_ok = bool(confidence_calibrated or not config.require_confidence_calibration)
    rul_gate_ok = bool(y_rul is None or not np.isfinite(y_rul).any()
                       or (config.max_rul_mae is not None and rul_mae is not None
                           and rul_mae <= config.max_rul_mae))
    amplitude_gate_ok = bool(y_amp is None or not np.isfinite(y_amp).any()
                             or (config.max_amplitude_mae is not None and amplitude_mae is not None
                                 and amplitude_mae <= config.max_amplitude_mae))
    anomaly_gate_ok = bool(y_anomaly is None or not np.isfinite(y_anomaly).any()
                           or (config.min_anomaly_auroc is not None and anomaly_auc is not None
                               and anomaly_auc >= config.min_anomaly_auroc))
    gate_passed = bool(accuracy >= config.min_val_accuracy and class_gate and calibration_gate_ok
                       and rul_gate_ok and amplitude_gate_ok and anomaly_gate_ok)
    gate_result = {
        "passed": gate_passed,
        "accuracy_ok": bool(accuracy >= config.min_val_accuracy),
        "per_class_recall_ok": bool(class_gate),
        "confidence_calibration_ok": calibration_gate_ok,
        "rul_gate_ok": rul_gate_ok,
        "amplitude_gate_ok": amplitude_gate_ok,
        "anomaly_gate_ok": anomaly_gate_ok,
        "min_val_accuracy": config.min_val_accuracy,
        "min_class_recall": config.min_class_recall,
        "min_class_support": config.min_class_support,
        "max_rul_mae": config.max_rul_mae,
        "max_amplitude_mae": config.max_amplitude_mae,
        "min_anomaly_auroc": config.min_anomaly_auroc,
    }

    output = Path(output_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output root already exists and is not empty: {output}")
    version_name = "M0" if gate_passed else "C0"
    version_dir = (output / "versions" / version_name) if gate_passed else (output / "candidates" / version_name)
    version_dir.mkdir(parents=True, exist_ok=True)
    ckpt = version_dir / "industrial_fault.pt"
    val_hash = hashlib.sha256(np.asarray(x_val, dtype="<f4").tobytes() +
                              np.asarray(val_labels_np, dtype="<i8").tobytes()).hexdigest()
    metadata = {
        "schema_version": 1, "task_type": "industrial_fault_diagnosis",
        "model_name": config.model_name, "model_version": "M0",
        "input_len": config.input_len, "n_features": config.n_features,
        "fault_classes": config.fault_classes, "train_config": asdict(config),
        "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
        "rul_scale": rul_scale, "amplitude_scale": amp_scale,
        "has_anomaly_head_targets": bool(y_anomaly is not None and np.isfinite(y_anomaly).any()),
        "has_rul_targets": bool(rul_train is not None and np.isfinite(rul_train).any()),
        "has_amplitude_targets": bool(amp_train is not None and np.isfinite(amp_train).any()),
        "temperature": temperature, "confidence_calibrated": confidence_calibrated,
        "calibration_id": calibration_id, "calibration_ece": calibration_error,
        "calibration_samples": calibration_samples, "validation_split_sha256": val_hash,
        "calibration_uses_disjoint_fit_eval": bool(x_cal_fit is not None and x_cal_eval is not None),
        "split_policy_asserted": config.split_policy,
        "rul_unit": config.rul_unit, "amplitude_unit": config.amplitude_unit,
        "release_gate": gate_result,
        "created_by": "diagnostic_agent.industrial_fault.train_industrial_fault_model",
    }
    torch.save({"metadata": metadata, "model_state": best_state}, ckpt)
    checkpoint_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    manifest = {
        "model_version": version_name, "model_name": config.model_name,
        "task_type": "industrial_fault_diagnosis", "checkpoint_filename": ckpt.name,
        "checkpoint_sha256": checkpoint_sha, "metadata": metadata,
        "release_eligible": gate_passed, "release_gate": gate_result,
        "validation_metrics": val_metrics,
    }
    (version_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if gate_passed:
        current = {"model_version": "M0", "activation_id": 1,
                   "activated_as_of": "bootstrap", "action": "bootstrap",
                   "task_type": "industrial_fault_diagnosis", "checkpoint_sha256": checkpoint_sha}
        (output / "current.json").write_text(json.dumps(current, indent=2), encoding="utf-8")
    (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"model_root": str(output), "model_version": version_name,
            "active": gate_passed, "checkpoint_sha256": checkpoint_sha,
            "release_gate": gate_result, "metadata": metadata,
            "validation_metrics": manifest["validation_metrics"],
            "candidate_path": str(version_dir) if not gate_passed else None}
