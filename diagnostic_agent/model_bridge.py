"""Adapters that connect the versioned LSTM/TCN forecast model to DiagnosticAgent.

The included ETTh1 model is a regression forecaster (24-step OT forecast), not a
fault classifier. This bridge preserves that task boundary: no fault label is
manufactured from the forecast. A missing classifier output triggers human review.
"""
from __future__ import annotations

import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np

from minimal_loop import FEATURES, H, L, VersionManager, load_csv


class VersionedForecastModelBridge:
    """Load the active immutable checkpoint and pass its forecast to the Agent."""

    def __init__(self, model_root: str | Path, *, device: str = "cpu"):
        self.model_root = Path(model_root)
        self.manager = VersionManager(self.model_root, device=device)

    def predict_context(self, csv_path: str | Path, *, equipment_id: str,
                        operating_condition: str = "dataset_replay",
                        episode_id: str | None = None) -> dict[str, Any]:
        values, times = load_csv(csv_path)
        if len(values) < L:
            raise ValueError(f"At least {L} observed rows are required; got {len(values)}")
        history = np.asarray(values[-L:], dtype=float)
        history_times = times[-L:]
        pointer = self.manager.current()
        model, manifest = self.manager.load(pointer["model_version"])
        forecast = model.predict_batch(history[None, :, :])[0]
        if forecast.shape != (H,) or not np.isfinite(forecast).all():
            raise ValueError("Active model returned an invalid forecast")
        prediction_time = times[-1] + timedelta(hours=1)
        target_times = [(prediction_time + timedelta(hours=i)).isoformat(timespec="seconds")
                        for i in range(H)]
        cfg = model.meta["train_config"]
        model_name = model.meta["model_name"]
        model_prediction = {
            "prediction_task": "time_series_forecasting",
            "prediction_kind": "regression",
            "target_name": model.meta.get("target", "OT"),
            "target_unit": "dataset_native",
            "forecast_horizon": H,
            "forecast_values": forecast.astype(float).tolist(),
            "forecast_times": target_times,
            "forecast_summary": {
                "first": float(forecast[0]), "last": float(forecast[-1]),
                "min": float(np.min(forecast)), "max": float(np.max(forecast)),
                "mean": float(np.mean(forecast)),
            },
            # The ETTh1 package model does not output fault class/confidence/OOD.
            # Explicit nulls prevent the Agent from interpreting regression as classification.
            "fault_type": None,
            "confidence": None,
            "confidence_calibrated": False,
            "ood_score": None,
            "drift_score": None,
            "input_quality": "valid",
            "train_config": cfg,
        }
        return {
            "episode_id": episode_id or f"MODEL-AGENT-{uuid.uuid4().hex[:12]}",
            "prediction_time": prediction_time.isoformat(timespec="seconds"),
            "equipment_id": equipment_id,
            "operating_condition": operating_condition,
            "task_type": "time_series_forecasting",
            "sensor_readings": {name: history[:, idx].astype(float).tolist()
                                for idx, name in enumerate(FEATURES)},
            "model": {
                "model_name": model_name,
                "model_version": pointer["model_version"],
                "activation_id": pointer["activation_id"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "model_task": "regression_forecast",
            },
            "model_prediction": model_prediction,
            "input_quality": "valid",
            "sensor_conflict": False,
            "source_data": {"path_name": Path(csv_path).name,
                            "history_rows": L,
                            "history_start": history_times[0].isoformat(timespec="seconds"),
                            "history_end": history_times[-1].isoformat(timespec="seconds")},
        }

    def run(self, agent: Any, csv_path: str | Path, *, equipment_id: str,
            operating_condition: str = "dataset_replay",
            episode_id: str | None = None) -> dict[str, Any]:
        context = self.predict_context(csv_path, equipment_id=equipment_id,
                                       operating_condition=operating_condition,
                                       episode_id=episode_id)
        report = agent.run(context)
        report["model_input_provenance"] = {
            "source": "VersionedForecastModelBridge",
            "model_root": str(self.model_root),
            "source_data": context["source_data"],
            "model_checkpoint_sha256": context["model"]["checkpoint_sha256"],
        }
        return report


class StructuredPredictorBridge:
    """Adapter for a user's real fault LSTM/TCN predictor.

    `predict_fn` must return a model-authored structured dict (e.g. fault_type,
    confidence, calibrated confidence, OOD/drift, RUL and fault amplitude). This
    class does not derive labels or confidence from raw sensor values.
    """

    def __init__(self, predict_fn: Callable[[np.ndarray], dict[str, Any]], *,
                 model_name: str, model_version: str, activation_id: int,
                 checkpoint_sha256: str, task_type: str = "industrial_fault_diagnosis"):
        if not callable(predict_fn):
            raise TypeError("predict_fn must be callable")
        if model_name not in {"LSTM", "TCN"}:
            raise ValueError("model_name must be LSTM or TCN")
        if not model_version or type(activation_id) is not int or activation_id < 0:
            raise ValueError("A model version and nonnegative activation_id are required")
        if not checkpoint_sha256:
            raise ValueError("checkpoint_sha256 is required for provenance")
        self.predict_fn = predict_fn
        self.model_meta = {"model_name": model_name, "model_version": model_version,
                           "activation_id": activation_id,
                           "checkpoint_sha256": checkpoint_sha256}
        self.task_type = task_type

    def predict_context(self, history: np.ndarray, *, equipment_id: str,
                        prediction_time: str, operating_condition: str,
                        sensor_readings: dict[str, Any] | None = None,
                        episode_id: str | None = None) -> dict[str, Any]:
        arr = np.asarray(history, dtype=float)
        if arr.ndim != 2 or len(arr) < 1 or not np.isfinite(arr).all():
            raise ValueError("history must be a finite 2-D numeric array")
        prediction = self.predict_fn(arr.copy())
        if not isinstance(prediction, dict):
            raise TypeError("predict_fn must return a structured dict")
        # A label is only passed through when the model itself returns it.
        return {
            "episode_id": episode_id or f"MODEL-AGENT-{uuid.uuid4().hex[:12]}",
            "prediction_time": prediction_time,
            "equipment_id": equipment_id,
            "operating_condition": operating_condition,
            "task_type": self.task_type,
            "sensor_readings": sensor_readings or {},
            "model": dict(self.model_meta),
            "model_prediction": dict(prediction),
        }

    def run(self, agent: Any, history: np.ndarray, **context_kwargs: Any) -> dict[str, Any]:
        return agent.run(self.predict_context(history, **context_kwargs))
