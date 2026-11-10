"""Inference adapter for the multi-transformer ETT oil-temperature checkpoints."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from models import LSTMModel, TCNModel

FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class ETTOilTemperatureBridge:
    """Load a trained ETT corpus checkpoint and call the shared DiagnosticAgent."""

    def __init__(self, model_dir: str | Path, *, device: str = "cpu"):
        import torch
        self.model_dir = Path(model_dir)
        self.device = device
        self.manifest = json.loads((self.model_dir / "manifest.json").read_text(encoding="utf-8"))
        self.checkpoint = self.model_dir / self.manifest["checkpoint"]
        if _sha256(self.checkpoint) != self.manifest["checkpoint_sha256"]:
            raise ValueError("ETT model checkpoint checksum mismatch")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self.metadata = payload["metadata"]
        if self.metadata.get("task") != "transformer_oil_temperature_forecasting":
            raise ValueError("Checkpoint is not an ETT oil-temperature forecaster")
        if self.metadata.get("features") != FEATURES or self.metadata.get("target") != "OT":
            raise ValueError("Checkpoint feature/target schema mismatch")
        if self.metadata.get("sample_interval_minutes") != self.manifest.get("sample_interval_minutes"):
            raise ValueError("Manifest/Checkpoint sampling interval mismatch")
        cfg = self.metadata["train_config"]
        self.model_name = self.metadata["model_name"]
        if self.model_name == "LSTM":
            self.net = LSTMModel(self.metadata["input_len"], len(FEATURES), self.metadata["pred_len"],
                                 hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
                                 dropout=cfg["dropout"])
        elif self.model_name == "TCN":
            self.net = TCNModel(self.metadata["input_len"], len(FEATURES), self.metadata["pred_len"],
                                channels=cfg["tcn_channels"], kernel_size=cfg["kernel_size"],
                                dropout=cfg["dropout"])
        else:
            raise ValueError(f"Unsupported ETT model: {self.model_name}")
        self.net.load_state_dict(payload["model_state"])
        self.net.to(device).eval()
        self.mean = np.asarray(payload["mean"], dtype=np.float32)
        self.scale = np.asarray(payload["scale"], dtype=np.float32)
        if self.mean.shape != (7,) or self.scale.shape != (7,) or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("Invalid ETT checkpoint normalization statistics")

    def predict(self, history: np.ndarray) -> np.ndarray:
        import torch
        arr = np.asarray(history, dtype=np.float32)
        expected = (self.metadata["input_len"], len(FEATURES))
        if arr.shape != expected or not np.isfinite(arr).all():
            raise ValueError(f"history must be finite with shape {expected}")
        x = (arr - self.mean) / self.scale
        with torch.no_grad():
            z = self.net(torch.from_numpy(x[None, ...]).to(self.device)).cpu().numpy()[0]
        pred = z * self.scale[6] + self.mean[6]
        if pred.shape != (self.metadata["pred_len"],) or not np.isfinite(pred).all():
            raise ValueError("Invalid oil-temperature forecast returned by model")
        return pred.astype(float)

    def predict_context(self, csv_path: str | Path, *, equipment_id: str,
                        episode_id: str | None = None, operating_condition: str = "dataset_replay") -> dict[str, Any]:
        path = Path(csv_path)
        rows, timestamps = [], []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            expected_cols = ["date", *FEATURES]
            if reader.fieldnames != expected_cols:
                raise ValueError(f"Expected columns {expected_cols}, got {reader.fieldnames}")
            for row in reader:
                timestamps.append(datetime.fromisoformat(row["date"]))
                rows.append([float(row[k]) for k in FEATURES])
        values = np.asarray(rows, dtype=np.float32)
        seq_len = int(self.metadata["input_len"])
        if len(values) < seq_len:
            raise ValueError(f"At least {seq_len} rows are required; got {len(values)}")
        interval = int(self.metadata["sample_interval_minutes"])
        for a, b in zip(timestamps[-seq_len:], timestamps[-seq_len+1:]):
            if int((b-a).total_seconds() // 60) != interval:
                raise ValueError("CSV sampling interval does not match the trained model")
        history, history_times = values[-seq_len:], timestamps[-seq_len:]
        forecast = self.predict(history)
        prediction_time = timestamps[-1] + timedelta(minutes=interval)
        target_times = [(prediction_time + timedelta(minutes=i*interval)).isoformat(timespec="seconds")
                        for i in range(len(forecast))]
        return {
            "episode_id": episode_id or f"ETT-{self.model_name}-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "prediction_time": prediction_time.isoformat(timespec="seconds"),
            "equipment_id": equipment_id,
            "operating_condition": operating_condition,
            "task_type": "transformer_oil_temperature_forecasting",
            "sensor_readings": {name: history[:, i].astype(float).tolist() for i, name in enumerate(FEATURES)},
            "model": {"model_name": self.model_name, "model_version": self.manifest["model_id"],
                      "activation_id": 0, "checkpoint_sha256": self.manifest["checkpoint_sha256"],
                      "model_task": "oil_temperature_regression", "frequency": self.manifest["frequency"]},
            "model_prediction": {
                "prediction_task": "transformer_oil_temperature_forecasting",
                "prediction_kind": "regression", "target_name": "OT", "target_unit": "degC",
                "forecast_horizon": len(forecast), "sample_interval_minutes": interval,
                "forecast_values": forecast.tolist(), "forecast_times": target_times,
                "forecast_summary": {"first": float(forecast[0]), "last": float(forecast[-1]),
                                      "min": float(np.min(forecast)), "max": float(np.max(forecast)),
                                      "mean": float(np.mean(forecast))},
                "fault_type": None, "confidence": None, "confidence_calibrated": False,
                "ood_score": None, "drift_score": None, "input_quality": "valid"},
            "input_quality": "valid", "sensor_conflict": False,
            "source_data": {"path_name": path.name, "history_rows": seq_len,
                             "history_start": history_times[0].isoformat(),
                             "history_end": history_times[-1].isoformat(),
                             "dataset_sources": self.metadata["data_sources"],
                             "sampling_interval_minutes": interval}}

    def run(self, agent: Any, csv_path: str | Path, **kwargs: Any) -> dict[str, Any]:
        context = self.predict_context(csv_path, **kwargs)
        report = agent.run(context)
        report["model_input_provenance"] = {"source": "ETTOilTemperatureBridge",
            "model_dir": str(self.model_dir), "model_checkpoint_sha256": self.manifest["checkpoint_sha256"],
            "source_data": context["source_data"]}
        return report
