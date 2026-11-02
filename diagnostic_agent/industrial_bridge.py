"""Versioned loader/bridge for independently trained industrial fault LSTM/TCN models."""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .industrial_fault import IndustrialFaultTrainConfig, build_fault_network


class VersionedIndustrialFaultBridge:
    def __init__(self, model_root: str | Path, *, device: str = "cpu",
                 runtime_signal_provider: Callable[[np.ndarray, dict[str, Any]], dict[str, Any]] | None = None):
        self.model_root = Path(model_root)
        self.device = device
        self.runtime_signal_provider = runtime_signal_provider
        self._load_active()

    def _load_active(self) -> None:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("PyTorch is required to load the industrial fault model") from exc
        pointer_path = self.model_root / "current.json"
        if not pointer_path.is_file():
            raise FileNotFoundError(f"Missing active model pointer: {pointer_path}")
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        version = pointer.get("model_version")
        version_dir = self.model_root / "versions" / str(version)
        manifest_path = version_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing model manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("release_eligible") is not True:
            raise ValueError("Industrial checkpoint has not passed its release gate")
        if self.manifest.get("task_type") != "industrial_fault_diagnosis":
            raise ValueError("Active checkpoint is not an industrial_fault_diagnosis model")
        checkpoint = version_dir / self.manifest["checkpoint_filename"]
        actual_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if actual_sha != self.manifest.get("checkpoint_sha256"):
            raise ValueError("Industrial fault checkpoint SHA-256 mismatch")
        if pointer.get("checkpoint_sha256") and pointer["checkpoint_sha256"] != actual_sha:
            raise ValueError("Active pointer and manifest checkpoint hashes differ")
        # weights_only=True avoids arbitrary pickle object loading.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.metadata = payload["metadata"]
        cfg_data = dict(self.metadata["train_config"])
        cfg_data["tcn_channels"] = tuple(cfg_data.get("tcn_channels", (32, 64, 64)))
        self.config = IndustrialFaultTrainConfig(**cfg_data)
        self.config.validate()
        self.net = build_fault_network(self.config)
        self.net.load_state_dict(payload["model_state"])
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device is unavailable")
        self.net.to(self.device).eval()
        self.mean = np.asarray(self.metadata["feature_mean"], dtype=np.float32)
        self.scale = np.asarray(self.metadata["feature_scale"], dtype=np.float32)
        if self.mean.shape != (self.config.n_features,) or self.scale.shape != self.mean.shape:
            raise ValueError("Checkpoint feature normalization shape mismatch")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("Checkpoint feature normalization values are invalid")
        self.pointer = pointer
        self.checkpoint = checkpoint

    def predict_context(self, history: np.ndarray, *, equipment_id: str,
                        prediction_time: str, operating_condition: str,
                        feature_names: list[str] | None = None,
                        sensor_readings: dict[str, Any] | None = None,
                        episode_id: str | None = None) -> dict[str, Any]:
        import torch
        arr = np.asarray(history, dtype=np.float32)
        expected = (self.config.input_len, self.config.n_features)
        if arr.ndim != 2 or arr.shape != expected:
            raise ValueError(f"history must have shape {expected}; got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError("history contains NaN or infinity")
        normalized = (arr - self.mean) / self.scale
        with torch.no_grad():
            outputs = self.net(torch.tensor(normalized[None], dtype=torch.float32, device=self.device))
            logits = outputs["fault_logits"][0]
            temperature = max(0.05, min(20.0, float(self.metadata.get("temperature", 1.0))))
            probabilities = torch.softmax(logits / temperature, dim=-1).cpu().numpy()
            class_index = int(np.argmax(probabilities))
            confidence = float(probabilities[class_index])
            anomaly_score = (float(torch.sigmoid(outputs["anomaly_logits"])[0].cpu())
                             if self.metadata.get("has_anomaly_head_targets") else None)
            rul = (float(outputs["rul_scaled"][0].cpu()) * float(self.metadata["rul_scale"])
                   if self.metadata.get("has_rul_targets") else None)
            amplitude = (float(outputs["amplitude_scaled"][0].cpu()) * float(self.metadata["amplitude_scale"])
                         if self.metadata.get("has_amplitude_targets") else None)
        runtime_signals = {}
        if self.runtime_signal_provider is not None:
            runtime_signals = self.runtime_signal_provider(arr.copy(), self.metadata) or {}
            if not isinstance(runtime_signals, dict):
                raise TypeError("runtime_signal_provider must return a dict")
        ood_score = runtime_signals.get("ood_score")
        drift_score = runtime_signals.get("drift_score")
        for key, value in (("ood_score", ood_score), ("drift_score", drift_score)):
            if value is not None and (isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{key} from runtime detector must be in [0,1] or None")
        prediction = {
            "prediction_task": "industrial_fault_diagnosis",
            "prediction_kind": "multi_task_classification_regression",
            "fault_type": self.metadata["fault_classes"][class_index],
            "fault_class_index": class_index,
            "fault_probabilities": {name: float(probabilities[i])
                                    for i, name in enumerate(self.metadata["fault_classes"])},
            "anomaly_score": anomaly_score,
            "confidence": confidence,
            "confidence_calibrated": bool(self.metadata.get("confidence_calibrated", False)),
            "calibration_id": self.metadata.get("calibration_id"),
            "ood_score": ood_score,
            "drift_score": drift_score,
            "rul_hours": rul,
            "rul_unit": self.metadata.get("rul_unit"),
            "fault_amplitude": amplitude,
            "fault_amplitude_unit": self.metadata.get("amplitude_unit"),
            "input_quality": runtime_signals.get("input_quality", "valid"),
        }
        if feature_names is not None and len(feature_names) != self.config.n_features:
            raise ValueError("feature_names count does not match checkpoint n_features")
        model = {
            "model_name": self.config.model_name,
            "model_version": self.pointer["model_version"],
            "activation_id": int(self.pointer["activation_id"]),
            "checkpoint_sha256": self.manifest["checkpoint_sha256"],
            "model_task": "industrial_fault_diagnosis",
            "calibration_id": self.metadata.get("calibration_id"),
        }
        return {
            "episode_id": episode_id or f"INDUSTRIAL-{uuid.uuid4().hex[:12]}",
            "prediction_time": prediction_time,
            "equipment_id": equipment_id,
            "operating_condition": operating_condition,
            "task_type": "industrial_fault_diagnosis",
            "sensor_readings": sensor_readings or {},
            "model": model,
            "model_prediction": prediction,
            "input_quality": prediction["input_quality"],
            "sensor_conflict": bool(runtime_signals.get("sensor_conflict", False)),
            "source_data": {"history_rows": self.config.input_len,
                            "n_features": self.config.n_features,
                            "feature_names": feature_names,
                            "model_root": str(self.model_root),
                            "checkpoint_sha256": self.manifest["checkpoint_sha256"]},
        }

    def run(self, agent: Any, history: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        context = self.predict_context(history, **kwargs)
        report = agent.run(context)
        report["model_input_provenance"] = {
            "source": "VersionedIndustrialFaultBridge",
            "model_root": str(self.model_root),
            "checkpoint_sha256": self.manifest["checkpoint_sha256"],
            "model_version": self.pointer["model_version"],
        }
        return report
