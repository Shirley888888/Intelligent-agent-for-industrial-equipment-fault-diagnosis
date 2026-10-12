"""Single-process feedback loop. No service, database, or scheduler.

NumPy Linear is a runnable reference backend; --backend torch uses the
unchanged repository LSTMModel. This module never calls train_gpu.main().
Every Tool receives only an observed prefix. Indices use half-open ranges.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np

FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
L, H, TARGET = 96, 24, 6


def require(condition: bool, message: str) -> None:
    # modify:explicit-validation - causal checks also run under python -O.
    if not condition:
        raise ValueError(message)


def check_observed(values: np.ndarray, times: list[datetime], as_of: datetime) -> None:
    # modify:prefix-contract - every observed row must be strictly before the clock.
    require(values.ndim == 2 and values.shape == (len(times), 7) and len(times) > 0,
            "Observed values must have shape [T,7] with T timestamps")
    require(np.isfinite(values).all(), "Observed values must be finite")
    require(all(t < as_of for t in times), "Future observations are forbidden")
    require(all(b-a == timedelta(hours=1) for a, b in zip(times, times[1:])),
            "Observed timestamps must be unique, sorted, hourly")


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def append_json(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iso(time: datetime) -> str:
    # ETTh1 dates are dataset-local timestamps, with no asserted UTC offset.
    return time.isoformat(timespec="seconds")


class FeedbackEpisode(TypedDict):
    episode_id: str
    prediction_time: str
    model_version: str
    activation_id: int
    origin_idx: int
    target_end_idx: int
    history: list[list[float]]
    history_times: list[str]
    forecast: list[float]
    target_times: list[str]
    truth: list[float] | None
    MAE: float | None
    feedback_time: str | None
    status: Literal["pending", "complete"]


@dataclass
class TrainConfig:
    backend: str = "numpy"
    model_name: str = "Linear"
    ridge: float = 1.0
    epochs: int = 10
    patience: int = 3
    lr: float = 0.001
    batch_size: int = 128
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    train_stride: int = 1
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    seed: int = 42
    device: str = "cpu"

    def validate(self) -> None:
        # modify:train-config - reject invalid settings before allocating a run directory.
        require((self.backend, self.model_name) in [("numpy", "Linear"), ("torch", "LSTM")],
                "Supported backend/model pairs are numpy/Linear and torch/LSTM")
        for name in ("epochs", "patience", "batch_size", "train_stride", "hidden_size", "num_layers"):
            value = getattr(self, name)
            require(type(value) is int and value > 0, f"{name} must be a positive integer")
        for name in ("lr", "ridge", "grad_clip"):
            require(np.isfinite(getattr(self, name)) and getattr(self, name) > 0,
                    f"{name} must be finite and positive")
        require(np.isfinite(self.weight_decay) and self.weight_decay >= 0,
                "weight_decay must be finite and nonnegative")
        require(np.isfinite(self.dropout) and 0 <= self.dropout < 1, "dropout must be in [0,1)")
        require(type(self.seed) is int and 0 <= self.seed < 2**32, "seed must be an unsigned 32-bit integer")
        require(isinstance(self.device, str) and bool(self.device), "device must be a nonempty string")


class EvalResult(TypedDict):
    MAE: float
    windows: int
    target_range: list[int]
    origins: list[int]


class ValidationMetrics(TypedDict):
    MAE: float
    unit: str
    windows: int
    best_epoch: int


class CandidateResult(TypedDict):
    candidate_id: str
    checkpoint_path: str
    checkpoint_sha256: str
    data_version: str
    trained_as_of: str
    train_config: dict
    validation_metrics: ValidationMetrics
    manifest: dict


class MonitorDecision(TypedDict):
    model_version: str
    activation_id: int
    n: int
    count: int
    baseline_mae: float
    recent_mae: float | None
    threshold: float
    episode_ids: list[str]
    checked_count: int
    evaluated_blocks: int
    degraded: bool
    streak: int
    trigger_retrain: bool
    reason: str
    as_of: str


class VersionPointer(TypedDict):
    model_version: str
    activation_id: int
    activated_as_of: str
    action: str
    baseline_mae: float


@dataclass
class GateConfig:
    recent_gain: float = 0.05
    old_tolerance: float = 0.02
    min_recent_gain: float = 0.01
    min_old_tolerance: float = 0.01

    def validate(self) -> None:
        require(all(np.isfinite(x) and x >= 0 for x in asdict(self).values()),
                "Gate margins must be finite and nonnegative")


class GateScores(TypedDict):
    allow_upgrade: bool
    recent_ok: bool
    historical_ok: bool
    recent_limit: float | None
    historical_limit: float | None
    incumbent_recent_mae: float | None
    candidate_recent_mae: float | None
    incumbent_historical_mae: float | None
    candidate_historical_mae: float | None


class GateResult(GateScores):
    incumbent_version: str
    incumbent_activation_id: int | None
    incumbent_checkpoint_sha256: str
    candidate_id: str
    checkpoint_sha256: str
    snapshot_sha256: str
    evaluated_as_of: str
    gate_config: dict
    recent: dict[str, EvalResult]
    historical: dict[str, EvalResult]


@dataclass
class DataVersion:
    data_version: str
    observed_end: int
    as_of: str
    train_ranges: list[tuple[int, int]]
    validation_range: tuple[int, int]
    historical_range: tuple[int, int]
    recent_range: tuple[int, int]
    snapshot_sha256: str


def snapshot_hash(values: np.ndarray, times: list[datetime]) -> str:
    h = hashlib.sha256(np.asarray(values, dtype="<f8").tobytes())
    h.update("\n".join(iso(x) for x in times).encode())
    return h.hexdigest()


def check_manifest(v: DataVersion, values: np.ndarray, times: list[datetime],
                   bootstrap: bool = False) -> None:
    # modify:split-validation - no duplicate fit rows or incomplete recent targets.
    check_observed(values, times, datetime.fromisoformat(v.as_of))
    bounds = v.train_ranges + [v.validation_range, v.historical_range, v.recent_range]
    require(all(len(r) == 2 and all(type(i) is int for i in r) for r in bounds),
            "All ranges must contain two integer indices")
    require(bool(v.train_ranges), "At least one training segment is required")
    ordered = sorted(v.train_ranges)
    require(all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:])),
            "Training segments must not overlap")
    require(len(values) == len(times) == v.observed_end, 'Invalid input or state: len(values) == len(times) == v.observed_end')
    require(len(values) and iso(times[-1]) < v.as_of, 'Invalid input or state: len(values) and iso(times[-1]) < v.as_of')
    require(v.snapshot_sha256 == snapshot_hash(values, times), 'Invalid input or state: v.snapshot_sha256 == snapshot_hash(values, times)')
    vs, ve = v.validation_range
    hs, he = v.historical_range
    rs, re = v.recent_range
    require(L <= vs < ve <= rs <= re <= v.observed_end, 'Invalid input or state: L <= vs < ve <= rs <= re <= v.observed_end')
    if bootstrap:
        require(ve == hs < he == rs, 'Invalid input or state: ve == hs < he == rs')
    else:
        require(L <= hs < he <= vs, 'Invalid input or state: L <= hs < he <= vs')
    for start, end in v.train_ranges:
        require(0 <= start < end <= vs, 'Invalid input or state: 0 <= start < end <= vs')
        require(end <= hs or start >= he, 'Invalid input or state: end <= hs or start >= he')
        require(end - start >= L + H, 'Invalid input or state: end - start >= L + H')
    require(ve - vs >= H and he - hs >= H, 'Invalid input or state: ve - vs >= H and he - hs >= H')
    require(re - rs >= H, "Recent gate must contain complete target windows")


def windows(values: np.ndarray, ranges: list[tuple[int, int]], stride: int,
            context_before: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training context stays inside each segment; holdout context may be earlier.
    All targets stay wholly inside the specified range, and history ends at p-1.
    """
    require(stride > 0, 'Invalid input or state: stride > 0')
    origins = []
    for start, end in ranges:
        first = max(L, start) if context_before else start + L
        origins.extend(range(first, end - H + 1, stride))
    if not origins:
        raise ValueError("Not enough complete windows")
    p = np.asarray(origins, dtype=int)
    x = np.stack([values[i-L:i] for i in p])
    y = np.stack([values[i:i+H, TARGET] for i in p])
    require(np.isfinite(x).all() and np.isfinite(y).all(), 'Invalid input or state: np.isfinite(x).all() and np.isfinite(y).all()')
    return x, y, p


class ForecastModel:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.checkpoint = Path(checkpoint)
        if self.checkpoint.suffix == ".npz":
            with np.load(self.checkpoint, allow_pickle=False) as f:
                self.meta = json.loads(str(f["metadata"]))
                self.mean, self.scale, self.w = f["mean"], f["scale"], f["w"]
            self.net = None
        elif self.checkpoint.suffix == ".pt":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            import torch
            from models import LSTMModel
            payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            # modify:meta-normalize - torch.save preserves tuples while JSON sidecars
            # store lists; round-trip so both backends compare equal.
            self.meta = json.loads(json.dumps(payload["metadata"]))
            self.mean = np.array(payload["mean"])
            self.scale = np.array(payload["scale"])
            cfg = self.meta["train_config"]
            self.net = LSTMModel(L, 7, H, hidden_size=cfg["hidden_size"],
                                 num_layers=cfg["num_layers"], dropout=cfg["dropout"])
            self.net.load_state_dict(payload["model_state"])
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("Requested CUDA device is unavailable")
            self.device = device
            self.net.to(device).eval()
        else:
            raise ValueError("Checkpoint extension must be .npz or .pt")
        # modify:checkpoint-contract - catch incompatible bundles before activation.
        require(self.meta.get("input_len") == L and self.meta.get("pred_len") == H,
                "Checkpoint input/forecast lengths are incompatible")
        cfg = TrainConfig(**self.meta["train_config"])
        cfg.validate()
        require(self.meta["model_name"] == cfg.model_name, "Checkpoint model/config mismatch")
        require((self.net is None) == (cfg.backend == "numpy"), "Checkpoint/backend mismatch")
        if self.net is None:
            require(self.w.shape == (L*7+1, H) and np.isfinite(self.w).all(),
                    "Linear weights must be finite and have shape [673,24]")
        require(self.meta["features"] == FEATURES and self.meta["target"] == "OT", 'Invalid input or state: self.meta["features"] == FEATURES and self.meta["target"] == "OT"')
        require(self.mean.shape == self.scale.shape == (7,), 'Invalid input or state: self.mean.shape == self.scale.shape == (7,)')
        require(np.isfinite(self.mean).all() and np.isfinite(self.scale).all(), 'Invalid input or state: np.isfinite(self.mean).all() and np.isfinite(self.scale).all()')
        require((self.scale > 0).all(), 'Invalid input or state: (self.scale > 0).all()')

    def predict_batch(self, history: np.ndarray) -> np.ndarray:
        require(history.ndim == 3 and history.shape[1:] == (L, 7), 'Invalid input or state: history.ndim == 3 and history.shape[1:] == (L, 7)')
        require(np.isfinite(history).all(), 'Invalid input or state: np.isfinite(history).all()')
        z = (history - self.mean) / self.scale
        if self.net is None:
            flat = z.reshape(len(z), -1)
            pred = np.concatenate([flat, np.ones((len(z), 1))], axis=1) @ self.w
        else:
            import torch
            batches = []
            with torch.no_grad():
                for i in range(0, len(z), 128):
                    t = torch.tensor(z[i:i+128], dtype=torch.float32, device=self.device)
                    batches.append(self.net(t).cpu().numpy())
            pred = np.concatenate(batches)
        result = pred * self.scale[TARGET] + self.mean[TARGET]
        require(result.shape == (len(z), H) and np.isfinite(result).all(), 'Invalid input or state: result.shape == (len(z), H) and np.isfinite(result).all()')
        return result


def predict_tool(model: ForecastModel, history: list[list[float]],
                 history_times: list[datetime], prediction_time: datetime,
                 model_version: str, activation_id: int, origin_idx: int) -> FeedbackEpisode:
    arr = np.asarray(history, dtype=float)
    require(type(origin_idx) is int and origin_idx >= L, "origin_idx must be at least 96")
    require(type(activation_id) is int and activation_id >= 0, "activation_id must be nonnegative")
    require(model_version.startswith("M") and model_version[1:].isdigit(), "Invalid model version")
    require(arr.shape == (L, 7) and len(history_times) == L, 'Invalid input or state: arr.shape == (L, 7) and len(history_times) == L')
    require(history_times[-1] < prediction_time, 'Invalid input or state: history_times[-1] < prediction_time')
    require(history_times[-1] + timedelta(hours=1) == prediction_time, 'Invalid input or state: history_times[-1] + timedelta(hours=1) == prediction_time')
    require(all(b-a == timedelta(hours=1) for a, b in zip(history_times, history_times[1:])), 'Invalid input or state: all(b-a == timedelta(hours=1) for a, b in zip(history_times, history_times[1:]))')
    target_times = [iso(prediction_time + timedelta(hours=i)) for i in range(H)]
    return dict(episode_id=f"E{origin_idx}", prediction_time=iso(prediction_time),
                model_version=model_version, activation_id=activation_id,
                origin_idx=origin_idx, target_end_idx=origin_idx+H,
                history=arr.tolist(), history_times=[iso(x) for x in history_times],
                forecast=model.predict_batch(arr[None])[0].tolist(),
                target_times=target_times, truth=None, MAE=None,
                feedback_time=None, status="pending")


def complete_feedback(ep: FeedbackEpisode, observed_values: np.ndarray,
                      observed_times: list[datetime], as_of: datetime) -> FeedbackEpisode:
    # modify:truth-contract - validate timestamps, forecast and maturity explicitly.
    check_observed(observed_values, observed_times, as_of)
    require(ep["status"] == "pending", 'Invalid input or state: ep["status"] == "pending"')
    start, end = ep["origin_idx"], ep["target_end_idx"]
    require(type(start) is int and type(end) is int and start >= L and end-start == H,
            "Episode target range must contain exactly 24 points")
    forecast = np.asarray(ep["forecast"], dtype=float)
    require(forecast.shape == (H,) and np.isfinite(forecast).all(), "Invalid Episode forecast")
    require(ep["target_times"][0] == ep["prediction_time"], "First truth time must equal prediction_time")
    require([iso(x) for x in observed_times[start-L:start]] == ep["history_times"],
            "Episode history timestamps do not match observed history")
    require(np.array_equal(np.asarray(ep["history"]), observed_values[start-L:start]),
            "Episode history values were changed")
    require(end <= len(observed_values) == len(observed_times), 'Invalid input or state: end <= len(observed_values) == len(observed_times)')
    require([iso(x) for x in observed_times[start:end]] == ep["target_times"], 'Invalid input or state: [iso(x) for x in observed_times[start:end]] == ep["target_times"]')
    require(observed_times[end-1] < as_of, 'Invalid input or state: observed_times[end-1] < as_of')
    truth = observed_values[start:end, TARGET]
    return {**ep, "truth": truth.tolist(),
            "MAE": float(np.mean(np.abs(truth - np.array(ep["forecast"])))),
            "feedback_time": iso(as_of), "status": "complete"}


class FeedbackMonitor:
    def __init__(self, n: int = 30, relative: float = 0.20, absolute: float = 0.10,
                 patience: int = 2, cooldown_episodes: int = 30):
        require(n > 0 and relative >= 0 and absolute >= 0 and patience > 0, 'Invalid input or state: n > 0 and relative >= 0 and absolute >= 0 and patience > 0')
        require(cooldown_episodes >= 0, 'Invalid input or state: cooldown_episodes >= 0')
        self.n, self.relative, self.absolute = n, relative, absolute
        self.patience, self.cooldown = patience, cooldown_episodes
        require(all(type(x) is int for x in (n, patience, cooldown_episodes)),
                "Monitor counts must be integers")
        require(np.isfinite(relative) and np.isfinite(absolute), "Monitor thresholds must be finite")
        self.reset()

    def reset(self) -> None:
        self.checked, self.streak, self.last_attempt_count = 0, 0, -10**9
        self.active_key = None

    def after_attempt(self, count: int) -> None:
        # Rejections and training failures also consume cooldown.
        self.last_attempt_count, self.streak = count, 0

    def run(self, episodes: list[FeedbackEpisode], model_version: str, activation_id: int,
            baseline_mae: float, as_of: str) -> MonitorDecision:
        # modify:rolling-monitor - stable output on every call; triggers use nonoverlapping blocks.
        datetime.fromisoformat(as_of)
        key = (model_version, activation_id)
        if self.active_key != key:
            self.reset()
            self.active_key = key
        require(np.isfinite(baseline_mae) and baseline_mae >= 0, 'Invalid input or state: np.isfinite(baseline_mae) and baseline_mae >= 0')
        mature = sorted([e for e in episodes if e["status"] == "complete"
                         and e["model_version"] == model_version
                         and e["activation_id"] == activation_id
                         and e["feedback_time"] <= as_of], key=lambda e: e["origin_idx"])
        for e in mature:
            require(e["target_times"][-1] < as_of and e["MAE"] is not None, 'Invalid input or state: e["target_times"][-1] < as_of and e["MAE"] is not None')
            require(np.isfinite(e["MAE"]) and e["MAE"] >= 0, "Feedback MAE must be finite and nonnegative")
        require(len({e["episode_id"] for e in mature}) == len(mature), "Duplicate feedback episodes")
        require(all(a["target_end_idx"] <= b["origin_idx"] for a, b in zip(mature, mature[1:])),
                "Monitor episodes must have nonoverlapping targets")
        count = len(mature)
        require(count >= self.checked, "Feedback history was truncated within one activation")
        threshold = baseline_mae + max(self.relative*baseline_mae, self.absolute)
        latest = mature[-self.n:] if count >= self.n else []
        base = dict(model_version=model_version, activation_id=activation_id,
                    n=self.n, count=count, baseline_mae=baseline_mae, as_of=as_of,
                    trigger_retrain=False,
                    recent_mae=float(np.mean([e["MAE"] for e in latest])) if latest else None,
                    episode_ids=[e["episode_id"] for e in latest], threshold=threshold,
                    checked_count=self.checked, evaluated_blocks=0,
                    degraded=False, streak=self.streak)
        if count < self.checked + self.n:
            return {**base, "reason": "waiting_for_next_nonoverlapping_block"}
        blocks = 0
        while self.checked + self.n <= count:
            block = mature[self.checked:self.checked+self.n]
            recent = float(np.mean([e["MAE"] for e in block]))
            degraded = recent > threshold
            self.streak = self.streak + 1 if degraded else 0
            self.checked += self.n
            blocks += 1
        trigger = self.streak >= self.patience and count-self.last_attempt_count >= self.cooldown
        return {**base, "checked_count": self.checked, "evaluated_blocks": blocks,
                "degraded": degraded, "streak": self.streak,
                "trigger_retrain": trigger,
                "reason": "degradation" if trigger else "hold"}


def train_candidate_tool(data_version: DataVersion, train_config: TrainConfig,
                         observed_values: np.ndarray, observed_times: list[datetime],
                         candidate_dir: str | Path, *, bootstrap: bool = False) -> CandidateResult:
    """Fit only train_ranges; select epochs only on validation_range.
    The old/recent gates are not passed into the optimiser or epoch selector.
    """
    check_manifest(data_version, observed_values, observed_times, bootstrap=bootstrap)
    cfg = train_config
    cfg.validate()
    require(cfg.train_stride > 0 and cfg.batch_size > 0, 'Invalid input or state: cfg.train_stride > 0 and cfg.batch_size > 0')
    require(cfg.epochs > 0 and cfg.patience > 0 and cfg.ridge > 0, 'Invalid input or state: cfg.epochs > 0 and cfg.patience > 0 and cfg.ridge > 0')
    out = Path(candidate_dir)
    out.mkdir(parents=True, exist_ok=False)
    train_rows = np.concatenate([observed_values[s:e] for s, e in data_version.train_ranges])
    mean, scale = train_rows.mean(axis=0), train_rows.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    x, y, _ = windows(observed_values, data_version.train_ranges, cfg.train_stride)
    vx, vy, _ = windows(observed_values, [data_version.validation_range], H, True)
    xz, yz = (x-mean)/scale, (y-mean[TARGET])/scale[TARGET]
    vz = (vx-mean)/scale
    meta = dict(schema_version=1, features=FEATURES, target="OT", input_len=L,
                pred_len=H, model_name=cfg.model_name, train_config=asdict(cfg),
                data_version=asdict(data_version), trained_as_of=data_version.as_of)
    best_epoch = 0
    if cfg.backend == "numpy":
        if cfg.model_name != "Linear":
            raise ValueError("NumPy reference implements Linear only")
        design = np.concatenate([xz.reshape(len(xz), -1), np.ones((len(xz), 1))], axis=1)
        penalty = np.eye(design.shape[1])*cfg.ridge
        penalty[-1, -1] = 0.0
        w = np.linalg.solve(design.T @ design + penalty, design.T @ yz)
        checkpoint = out/"checkpoint.npz"
        np.savez(checkpoint, mean=mean, scale=scale, w=w, metadata=json.dumps(meta))
    elif cfg.backend == "torch":
        if cfg.model_name != "LSTM":
            raise ValueError("Torch adapter preserves the deployed LSTM only")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from models import LSTMModel
        from torch.utils.data import DataLoader, TensorDataset
        torch.manual_seed(cfg.seed)
        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device is unavailable")
        torch.use_deterministic_algorithms(True)
        net = LSTMModel(L, 7, H, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers,
                        dropout=cfg.dropout).to(cfg.device)
        loader = DataLoader(TensorDataset(torch.tensor(xz, dtype=torch.float32),
                                         torch.tensor(yz, dtype=torch.float32)),
                            batch_size=cfg.batch_size, shuffle=True, num_workers=0)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        best, bad, state = float("inf"), 0, None
        for epoch in range(1, cfg.epochs+1):
            net.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(net(xb.to(cfg.device)), yb.to(cfg.device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                opt.step()
            net.eval()
            with torch.no_grad():
                pred = np.concatenate([net(torch.tensor(vz[i:i+128], dtype=torch.float32,
                                                        device=cfg.device)).cpu().numpy()
                                       for i in range(0, len(vz), 128)])
            score = float(np.mean(np.abs(pred*scale[TARGET]+mean[TARGET]-vy)))
            if not np.isfinite(score):
                raise ValueError("Nonfinite candidate validation MAE")
            if score < best:
                best, bad, best_epoch = score, 0, epoch
                state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
            if bad >= cfg.patience:
                break
        checkpoint = out/"checkpoint.pt"
        torch.save(dict(model_state=state, mean=mean.tolist(), scale=scale.tolist(), metadata=meta), checkpoint)
    else:
        raise ValueError("Unknown backend")
    model = ForecastModel(checkpoint, cfg.device)
    validation_mae = float(np.mean(np.abs(model.predict_batch(vx)-vy)))
    require(np.isfinite(validation_mae), "Validation MAE must be finite")
    result = dict(candidate_id=out.name, checkpoint_path=str(checkpoint.resolve()),
                  checkpoint_sha256=digest(checkpoint), data_version=data_version.data_version,
                  trained_as_of=data_version.as_of, train_config=asdict(cfg),
                  validation_metrics=dict(MAE=validation_mae, unit="degC", windows=len(vy),
                                          best_epoch=best_epoch), manifest=asdict(data_version))
    write_json(out/"manifest.json", result)
    return result


def eval_range(model: ForecastModel, values: np.ndarray, target_range: tuple[int, int]) -> EvalResult:
    require(L <= target_range[0] < target_range[1] <= len(values), "Evaluation range is outside observed data")
    x, y, origins = windows(values, [target_range], H, True)
    pred = model.predict_batch(x)
    return dict(MAE=float(np.mean(np.abs(pred-y))), windows=len(y),
                target_range=list(target_range), origins=origins.tolist())


def model_gate(incumbent_recent: float, candidate_recent: float,
               incumbent_old: float, candidate_old: float,
               recent_gain: float = 0.05, old_tolerance: float = 0.02,
               min_recent_gain: float = 0.01, min_old_tolerance: float = 0.01) -> GateScores:
    GateConfig(recent_gain, old_tolerance, min_recent_gain, min_old_tolerance).validate()
    scores = [incumbent_recent, candidate_recent, incumbent_old, candidate_old]
    valid = all(np.isfinite(x) and x >= 0 for x in scores)
    recent_limit = incumbent_recent - max(recent_gain*incumbent_recent, min_recent_gain)
    old_limit = incumbent_old + max(old_tolerance*incumbent_old, min_old_tolerance)
    recent_ok = valid and candidate_recent < incumbent_recent and candidate_recent <= recent_limit
    old_ok = valid and candidate_old <= old_limit
    # modify:finite-gate-output - invalid metrics reject and remain valid JSON (null).
    safe = lambda x: float(x) if np.isfinite(x) and x >= 0 else None
    return dict(allow_upgrade=bool(recent_ok and old_ok), recent_ok=bool(recent_ok),
                historical_ok=bool(old_ok), recent_limit=float(recent_limit) if np.isfinite(recent_limit) else None,
                historical_limit=safe(old_limit),
                incumbent_recent_mae=safe(incumbent_recent), candidate_recent_mae=safe(candidate_recent),
                incumbent_historical_mae=safe(incumbent_old), candidate_historical_mae=safe(candidate_old))


def validate_candidate(candidate: CandidateResult, device: str = "cpu") -> ForecastModel:
    # modify:candidate-binding - sidecar claims must match the actual checkpoint.
    require(digest(Path(candidate["checkpoint_path"])) == candidate["checkpoint_sha256"],
            "Candidate checkpoint checksum mismatch")
    model = ForecastModel(candidate["checkpoint_path"], device)
    require(model.meta["train_config"] == candidate["train_config"], "Candidate configuration mismatch")
    require(model.meta["data_version"] == json.loads(json.dumps(candidate["manifest"])), "Candidate data manifest mismatch")
    require(candidate["manifest"]["data_version"] == candidate["data_version"] and
            candidate["manifest"]["as_of"] == candidate["trained_as_of"], "Candidate provenance mismatch")
    require(np.isfinite(candidate["validation_metrics"]["MAE"]) and
            candidate["validation_metrics"]["MAE"] >= 0, "Invalid validation MAE")
    model.predict_batch(np.zeros((1, L, 7)))
    return model


def model_gate_tool(candidate: CandidateResult, incumbent: ForecastModel, incumbent_version: str,
                    observed_values: np.ndarray, observed_times: list[datetime],
                    device: str = "cpu", *, incumbent_activation_id: int | None = None,
                    gate_config: GateConfig | None = None) -> GateResult:
    v = DataVersion(**candidate["manifest"])
    check_manifest(v, observed_values, observed_times)
    require(digest(Path(candidate["checkpoint_path"])) == candidate["checkpoint_sha256"], 'Invalid input or state: digest(Path(candidate["checkpoint_path"])) == candidate["checkpoint_sha256"]')
    new = validate_candidate(candidate, device)
    cfg = gate_config or GateConfig()
    cfg.validate()
    ir, cr = eval_range(incumbent, observed_values, v.recent_range), eval_range(new, observed_values, v.recent_range)
    io, co = eval_range(incumbent, observed_values, v.historical_range), eval_range(new, observed_values, v.historical_range)
    return {**model_gate(ir["MAE"], cr["MAE"], io["MAE"], co["MAE"], **asdict(cfg)),
            "incumbent_version": incumbent_version, "candidate_id": candidate["candidate_id"],
            "incumbent_activation_id": incumbent_activation_id,
            "incumbent_checkpoint_sha256": digest(incumbent.checkpoint),
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "snapshot_sha256": v.snapshot_sha256, "gate_config": asdict(cfg),
            "evaluated_as_of": v.as_of, "recent": {"incumbent": ir, "candidate": cr},
            "historical": {"incumbent": io, "candidate": co}}


class VersionManager:
    """Immutable model directories + atomically replaced current.json; single writer.
    Each activation increments activation_id, including rollbacks to the same version.
    """
    def __init__(self, root: str | Path, device: str = "cpu"):
        self.root, self.device = Path(root), device
        (self.root/"versions").mkdir(parents=True, exist_ok=True)

    def current(self) -> VersionPointer:
        return json.loads((self.root/"current.json").read_text(encoding="utf-8"))

    def list_versions(self) -> list[dict]:
        # modify:version-inventory - review available targets before a one-command rollback.
        current = self.current() if (self.root/"current.json").exists() else None
        versions = sorted((p.name for p in (self.root/"versions").iterdir()
                           if p.is_dir() and p.name.startswith("M") and p.name[1:].isdigit()),
                          key=lambda name: int(name[1:]))
        result = []
        for version in versions:
            _, manifest = self.load(version)
            result.append(dict(model_version=version, model_name=manifest["train_config"]["model_name"],
                               baseline_mae=manifest["baseline_mae"], activated_as_of=manifest["activated_as_of"],
                               current=bool(current and current["model_version"] == version)))
        return result

    def load(self, version: str) -> tuple[ForecastModel, dict]:
        require(version.startswith("M") and version[1:].isdigit(), 'Invalid input or state: version.startswith("M") and version[1:].isdigit()')
        path = self.root/"versions"/version
        manifest = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
        require(manifest["model_version"] == version, "Version manifest identity mismatch")
        require(np.isfinite(manifest["baseline_mae"]) and manifest["baseline_mae"] >= 0,
                "Saved version baseline must be finite and nonnegative")
        require(manifest["checkpoint_filename"] == f'{version}_checkpoint{Path(manifest["checkpoint_filename"]).suffix}',
                "Version checkpoint filename is invalid")
        cp = path/manifest["checkpoint_filename"]
        if digest(cp) != manifest["checkpoint_sha256"]:
            raise ValueError("Checkpoint checksum mismatch; current pointer unchanged")
        model = ForecastModel(cp, self.device)  # shape/scaler validated before activation
        require(model.meta["train_config"] == manifest["train_config"] and
                model.meta["data_version"] == manifest["manifest"], "Version sidecar/checkpoint mismatch")
        model.predict_batch(np.zeros((1, L, 7)))  # finite smoke inference
        return model, manifest

    def publish(self, candidate: CandidateResult, gate: GateResult | None, as_of: str,
                baseline_mae: float) -> VersionPointer:
        # modify:publication-binding - stale gates after rollback must never activate.
        datetime.fromisoformat(as_of)
        validate_candidate(candidate, self.device)
        versions = [int(p.name[1:]) for p in (self.root/"versions").iterdir()
                    if p.is_dir() and p.name.startswith("M") and p.name[1:].isdigit()]
        current = self.current() if (self.root/"current.json").exists() else None
        if current is not None:
            require(gate and gate["allow_upgrade"], 'Invalid input or state: gate and gate["allow_upgrade"]')
            require(gate["incumbent_version"] == current["model_version"], 'Invalid input or state: gate["incumbent_version"] == current["model_version"]')
            require(gate["candidate_id"] == candidate["candidate_id"], 'Invalid input or state: gate["candidate_id"] == candidate["candidate_id"]')
            require(gate["checkpoint_sha256"] == candidate["checkpoint_sha256"], 'Invalid input or state: gate["checkpoint_sha256"] == candidate["checkpoint_sha256"]')
            require(gate["evaluated_as_of"] <= as_of, 'Invalid input or state: gate["evaluated_as_of"] <= as_of')
            require(as_of >= current["activated_as_of"], "Activation clock cannot move backwards")
            require(gate["incumbent_activation_id"] == current["activation_id"], "Stale gate activation")
            incumbent, _ = self.load(current["model_version"])
            require(gate["incumbent_checkpoint_sha256"] == digest(incumbent.checkpoint), "Stale incumbent weights")
            require(gate["snapshot_sha256"] == candidate["manifest"]["snapshot_sha256"], "Gate/snapshot mismatch")
            require(gate["evaluated_as_of"] == candidate["trained_as_of"], "Gate/candidate clock mismatch")
            calculated = model_gate(gate["incumbent_recent_mae"], gate["candidate_recent_mae"],
                                    gate["incumbent_historical_mae"], gate["candidate_historical_mae"],
                                    **gate["gate_config"])
            require(calculated["allow_upgrade"] and gate["recent_ok"] and gate["historical_ok"],
                    "Both quantitative gate conditions must pass")
            require(baseline_mae == gate["candidate_recent_mae"], "New baseline must equal the gate's recent MAE")
        else:
            require(gate is None and not versions, 'Invalid input or state: gate is None and not versions')
        require(candidate["trained_as_of"] <= as_of, 'Invalid input or state: candidate["trained_as_of"] <= as_of')
        require(np.isfinite(baseline_mae) and baseline_mae >= 0, 'Invalid input or state: np.isfinite(baseline_mae) and baseline_mae >= 0')
        cp = Path(candidate["checkpoint_path"])
        require(digest(cp) == candidate["checkpoint_sha256"], 'Invalid input or state: digest(cp) == candidate["checkpoint_sha256"]')
        version = f"M{max(versions, default=-1)+1}"
        destination = self.root/"versions"/version
        filename = f"{version}_checkpoint{cp.suffix}"
        manifest = {**candidate, "model_version": version, "checkpoint_filename": filename,
                    "parent_version": current["model_version"] if current else None,
                    "activated_as_of": as_of, "baseline_mae": baseline_mae}
        # modify:atomic-version - only a complete copied bundle obtains an M number.
        with tempfile.TemporaryDirectory(prefix="publish_", dir=self.root) as folder:
            staged = Path(folder)/version
            staged.mkdir()
            shutil.copy2(cp, staged/filename)
            require(digest(staged/filename) == candidate["checkpoint_sha256"], "Copied checkpoint checksum mismatch")
            write_json(staged/"manifest.json", manifest)
            write_json(staged/"gate.json", gate or {"bootstrap": True})
            os.replace(staged, destination)
        try:
            return self.activate(version, as_of, "bootstrap" if current is None else "upgrade")
        except Exception:
            # modify:failed-publication - remove only this call's unactivated bundle.
            # activate has no fallible I/O after its pointer commit.
            shutil.rmtree(destination)
            raise

    def activate(self, version: str, as_of: str, action: str = "rollback") -> VersionPointer:
        datetime.fromisoformat(as_of)
        require(action in ("rollback", "bootstrap", "upgrade"), "Invalid activation action")
        self.load(version)
        manifest = json.loads((self.root/"versions"/version/"manifest.json").read_text(encoding="utf-8"))
        require(manifest["activated_as_of"] <= as_of, 'Invalid input or state: manifest["activated_as_of"] <= as_of')
        previous = self.current() if (self.root/"current.json").exists() else None
        require(previous is None or previous["activated_as_of"] <= as_of,
                "Rollback clock cannot precede the current activation")
        result = dict(model_version=version,
                      activation_id=previous["activation_id"]+1 if previous else 0,
                      activated_as_of=as_of, action=action,
                      baseline_mae=manifest["baseline_mae"])
        # modify:activation-commit - single writer; pointer replacement commits last.
        # On ordinary I/O exceptions, restore even a partially appended event.
        # This is not a multi-file transaction against process/power failure.
        pointer = self.root/"current.json"
        journal = self.root/"version_events.jsonl"
        had_journal = journal.exists()
        journal_size = journal.stat().st_size if had_journal else 0
        try:
            append_json(journal, {**result, "previous": previous})
            write_json(pointer, result)
        except Exception as exc:
            try:
                if had_journal:
                    with journal.open("r+b") as f:
                        f.truncate(journal_size)
                else:
                    journal.unlink(missing_ok=True)
                pointer.with_suffix(pointer.suffix + ".tmp").unlink(missing_ok=True)
            except Exception as cleanup_exc:
                raise RuntimeError(
                    f"Activation failed before pointer commit ({exc}); "
                    f"event/temp cleanup also failed: {cleanup_exc}"
                ) from exc
            raise
        return result


def load_csv(path: str | Path) -> tuple[np.ndarray, list[datetime]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    times = [datetime.fromisoformat(r["date"]) for r in rows]
    values = np.array([[float(r[c]) for c in FEATURES] for r in rows], dtype=float)
    require(all(b-a == timedelta(hours=1) for a, b in zip(times, times[1:])), 'Invalid input or state: all(b-a == timedelta(hours=1) for a, b in zip(times, times[1:]))')
    require(np.isfinite(values).all(), 'Invalid input or state: np.isfinite(values).all()')
    return values, times


def make_version(values: np.ndarray, times: list[datetime], as_of: datetime,
                 ranges: list[tuple[int, int]], validation: tuple[int, int],
                 historical: tuple[int, int], recent: tuple[int, int]) -> DataVersion:
    sh = snapshot_hash(values, times)
    return DataVersion(f"D{len(values)}_{sh[:12]}", len(values), iso(as_of), ranges,
                       validation, historical, recent, sh)


def simulate(values: np.ndarray, times: list[datetime], out: str | Path,
             config: TrainConfig | None = None, monitor_n: int = 30,
             patience: int = 2, exercise_at: int | None = None,
             max_episodes: int | None = None, train_end: int = 12194,
             online_start: int = 14807, monitor_relative: float = 0.20,
             monitor_absolute: float = 0.10, cooldown_episodes: int | None = None,
             gate_config: GateConfig | None = None) -> dict:
    config = config or TrainConfig()
    # modify:config-entry - tune on development data, then freeze the replay settings.
    config.validate()
    gate_config = gate_config or GateConfig()
    gate_config.validate()
    cooldown = monitor_n if cooldown_episodes is None else cooldown_episodes
    monitor = FeedbackMonitor(n=monitor_n, relative=monitor_relative, absolute=monitor_absolute,
                              patience=patience, cooldown_episodes=cooldown)
    require(max_episodes is None or type(max_episodes) is int and max_episodes > 0,
            "max_episodes must be positive")
    require(exercise_at is None or type(exercise_at) is int and exercise_at >= monitor_n,
            "exercise_at must be at least monitor_n")
    require(exercise_at is None or max_episodes is None or exercise_at <= max_episodes,
            "exercise_at exceeds max_episodes")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=False)  # do not overwrite previous experiments
    require(0 < train_end < online_start < len(values), 'Invalid input or state: 0 < train_end < online_start < len(values)')
    require(len(values) == len(times), 'Invalid input or state: len(values) == len(times)')
    require(all(b-a == timedelta(hours=1) for a, b in zip(times, times[1:])), 'Invalid input or state: all(b-a == timedelta(hours=1) for a, b in zip(times, times[1:]))')
    val_end = train_end + 720
    baseline_start = online_start - 720
    anchor = (val_end, baseline_start)
    require(anchor[1]-anchor[0] >= H, 'Invalid input or state: anchor[1]-anchor[0] >= H')
    manager = VersionManager(root, config.device)
    initial_values, initial_times = values[:online_start].copy(), times[:online_start]
    now = initial_times[-1] + timedelta(hours=1)
    m0_version = make_version(initial_values, initial_times, now, [(0, train_end)],
                              (train_end, val_end), (val_end, baseline_start),
                              (baseline_start, online_start))
    m0 = train_candidate_tool(m0_version, config, initial_values, initial_times,
                              root/"candidates"/"C0", bootstrap=True)
    model = ForecastModel(m0["checkpoint_path"], config.device)
    baseline = eval_range(model, initial_values, (baseline_start, online_start))
    write_json(root/"baseline_M0.json", baseline)
    state = manager.publish(m0, None, iso(now), baseline["MAE"])
    write_json(root/"run_config.json", dict(train_config=asdict(config), gate_config=asdict(gate_config),
                                          monitor=dict(n=monitor_n, relative=monitor_relative,
                                                       absolute=monitor_absolute, patience=patience,
                                                       cooldown_episodes=cooldown),
                                          train_end=train_end, online_start=online_start))
    pending, complete, events = [], [], []
    attempt = 0
    stop = min(len(values), online_start + max_episodes*H) if max_episodes is not None else len(values)
    for t in range(online_start, stop+1):
        now = times[t-1] + timedelta(hours=1)
        observed, observed_times = values[:t].copy(), times[:t]
        require(observed_times[-1] < now, 'Invalid input or state: observed_times[-1] < now')
        matured = [ep for ep in pending if ep["target_end_idx"] <= t]
        pending = [ep for ep in pending if ep["target_end_idx"] > t]
        for ep in matured:
            finished = complete_feedback(ep, observed, observed_times, now)
            complete.append(finished)
            append_json(root/"feedback.jsonl", finished)
        decision = monitor.run(complete, state["model_version"], state["activation_id"],
                               state["baseline_mae"], iso(now))
        active_eps = [ep for ep in complete if ep["activation_id"] == state["activation_id"]]
        exercise = exercise_at is not None and len(complete) == exercise_at and bool(matured)
        if decision["evaluated_blocks"] or exercise:
            append_json(root/"monitor.jsonl", {**decision, "engineering_exercise": exercise})
        if decision["trigger_retrain"] or exercise:
            monitor.after_attempt(len(active_eps))
            attempt += 1
            event = dict(as_of=iso(now), engineering_exercise=exercise,
                         trigger=decision, candidate_id=f"C{attempt}")
            try:
                # modify:failure-retention - range failures also leave the active model intact.
                require(len(active_eps) >= monitor_n, "Candidate requires enough mature feedback in this activation")
                recent_eps = active_eps[-monitor_n:]
                recent = (recent_eps[0]["origin_idx"], recent_eps[-1]["target_end_idx"])
                validation = (recent[0]-720, recent[0])
                require(validation[0] >= anchor[1], "Not enough past data for isolated candidate validation")
                ranges = [(0, anchor[0])]
                if validation[0]-anchor[1] >= L+H:
                    ranges.append((anchor[1], validation[0]))
                version = make_version(observed, observed_times, now, ranges, validation, anchor, recent)
                candidate = train_candidate_tool(version, config, observed, observed_times,
                                                  root/"candidates"/f"C{attempt}")
                gate = model_gate_tool(candidate, model, state["model_version"],
                                       observed, observed_times, config.device,
                                       incumbent_activation_id=state["activation_id"], gate_config=gate_config)
                event["gate"] = gate
                write_json(root/"candidates"/f"C{attempt}"/"gate.json", gate)
                if gate["allow_upgrade"]:
                    # modify:prepare-before-commit - no checkpoint I/O after activation.
                    prepared_model = ForecastModel(candidate["checkpoint_path"], config.device)
                    prepared_model.predict_batch(np.zeros((1, L, 7)))
                    state = manager.publish(candidate, gate, iso(now), gate["candidate_recent_mae"])
                    model = prepared_model
                    monitor.reset()
                    event["new_version"] = state["model_version"]
                else:
                    event["new_version"] = None
            except Exception as exc:
                # Failed candidate never changes the active model; retain the failure.
                event["error"] = f"{type(exc).__name__}: {exc}"
            events.append(event)
            append_json(root/"retrain_events.jsonl", event)
        if t < stop and (t-online_start) % H == 0:
            ep = predict_tool(model, observed[t-L:t].tolist(), observed_times[t-L:t],
                              now, state["model_version"], state["activation_id"], t)
            pending.append(ep)
            append_json(root/"predictions.jsonl", ep)
    mae = float(np.mean([e["MAE"] for e in complete])) if complete else None
    summary = dict(backend=config.backend, model_name=config.model_name,
                   complete_episodes=len(complete), pending_episodes=len(pending),
                   baseline_mae=baseline["MAE"], prequential_mae=mae,
                   retrain_attempts=len(events), upgrades=sum("new_version" in e and e["new_version"] is not None for e in events),
                   candidate_errors=[e["error"] for e in events if "error" in e],
                   current=state, observed_end=stop, anchor=list(anchor),
                   final_as_of=iso(now), engineering_exercise_at=exercise_at)
    write_json(root/"summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    subs = ap.add_subparsers(dest="command", required=True)
    sim = subs.add_parser("simulate")
    sim.add_argument("--csv", default=str(Path(__file__).with_name("ETTh1.csv")))
    sim.add_argument("--out", required=True)
    # modify:full-cli - JSON configuration plus explicit command-line overrides.
    sim.add_argument("--train-config", help="TrainConfig JSON file")
    sim.add_argument("--gate-config", help="GateConfig JSON file")
    sim.add_argument("--backend", choices=["numpy", "torch"])
    sim.add_argument("--device")
    sim.add_argument("--epochs", type=int)
    sim.add_argument("--train-patience", type=int)
    for name in ("batch-size", "train-stride", "hidden-size", "num-layers", "seed"):
        sim.add_argument("--"+name, type=int)
    for name in ("lr", "weight-decay", "grad-clip", "dropout", "ridge"):
        sim.add_argument("--"+name, type=float)
    sim.add_argument("--monitor-n", type=int, default=30)
    sim.add_argument("--patience", type=int, default=2)
    sim.add_argument("--monitor-relative", type=float, default=0.20)
    sim.add_argument("--monitor-absolute", type=float, default=0.10)
    sim.add_argument("--cooldown-episodes", type=int)
    for name in ("recent-gain", "old-tolerance", "min-recent-gain", "min-old-tolerance"):
        sim.add_argument("--"+name, type=float)
    sim.add_argument("--exercise-at", type=int)
    sim.add_argument("--max-episodes", type=int)
    rb = subs.add_parser("rollback")
    rb.add_argument("--out", required=True)
    rb.add_argument("--version", required=True)
    rb.add_argument("--device", default="cpu")
    listing = subs.add_parser("versions")
    listing.add_argument("--out", required=True)
    listing.add_argument("--device", default="cpu")
    catalog = subs.add_parser("tools", help="Print complete Tool signatures and I/O schemas")
    catalog.add_argument("--out", help="Optional JSON output path")
    args = ap.parse_args()
    if args.command == "simulate":
        values, times = load_csv(args.csv)
        settings = json.loads(Path(args.train_config).read_text(encoding="utf-8")) if args.train_config else {}
        if args.backend is not None:
            settings.update(backend=args.backend, model_name="LSTM" if args.backend == "torch" else "Linear")
        elif "backend" in settings and "model_name" not in settings:
            settings["model_name"] = "LSTM" if settings["backend"] == "torch" else "Linear"
        for name in asdict(TrainConfig()):
            value = getattr(args, "train_patience" if name == "patience" else name, None)
            if value is not None:
                settings[name] = value
        cfg = TrainConfig(**settings)
        margins = json.loads(Path(args.gate_config).read_text(encoding="utf-8")) if args.gate_config else {}
        for name in asdict(GateConfig()):
            if getattr(args, name) is not None:
                margins[name] = getattr(args, name)
        result = simulate(values, times, args.out, cfg, args.monitor_n, args.patience,
                          args.exercise_at, args.max_episodes, monitor_relative=args.monitor_relative,
                          monitor_absolute=args.monitor_absolute, cooldown_episodes=args.cooldown_episodes,
                          gate_config=GateConfig(**margins))
    elif args.command == "rollback":
        manager = VersionManager(args.out, args.device)
        current = manager.current()
        # CLI acts at the recorded final simulated clock, not wall-clock ETTh1 time.
        summary = json.loads((Path(args.out)/"summary.json").read_text(encoding="utf-8"))
        result = manager.activate(args.version, max(current["activated_as_of"], summary["final_as_of"]), "rollback")
    elif args.command == "versions":
        result = VersionManager(args.out, args.device).list_versions()
    else:
        from tool_catalog import tool_catalog
        result = tool_catalog()
        if args.out:
            write_json(Path(args.out), result)
            result = {"saved_to": str(Path(args.out).resolve()), "tool_count": len(result["tools"])}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
