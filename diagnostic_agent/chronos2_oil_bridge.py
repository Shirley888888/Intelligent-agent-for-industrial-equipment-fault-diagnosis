"""Bridge a fine-tuned Chronos-2 model into the shared oil-temperature Agent."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from train_ett_corpus import COLS, FREQ_MINUTES, load_series


def sha256_tree(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        h.update(len(rel).to_bytes(8, "big")); h.update(rel)
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


class Chronos2OilTemperatureBridge:
    """Load a local Chronos-2 fine-tune artifact and pass quantile forecasts to DiagnosticAgent."""

    def __init__(self, model_dir: str | Path, *, device_map: str = "auto", pipeline: Any = None):
        self.model_dir = Path(model_dir)
        self.manifest = json.loads((self.model_dir / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("task") != "transformer_oil_temperature_forecasting":
            raise ValueError("Model manifest is not for transformer oil-temperature forecasting")
        self.artifact_dir = self.model_dir / self.manifest.get("model_artifact", "finetuned-ckpt")
        if not self.artifact_dir.is_dir():
            raise FileNotFoundError(f"Fine-tuned Chronos model not found: {self.artifact_dir}")
        actual = sha256_tree(self.artifact_dir)
        if actual != self.manifest.get("model_artifact_sha256_tree"):
            raise ValueError("Chronos-2 model artifact checksum mismatch")
        if pipeline is None:
            try:
                from chronos import Chronos2Pipeline
            except ImportError as exc:
                raise RuntimeError("Install requirements_chronos.txt before loading Chronos-2") from exc
            pipeline = Chronos2Pipeline.from_pretrained(str(self.artifact_dir), device_map=device_map)
        self.pipeline = pipeline
        self.frequency = self.manifest["frequency"]
        self.interval = int(self.manifest["sample_interval_minutes"])
        self.input_len = int(self.manifest["context_length"])
        self.pred_len = int(self.manifest["prediction_length"])
        self.quantile_levels = list(self.manifest.get("quantile_levels", [0.1, 0.5, 0.9]))
        self.model_id = self.manifest.get("model_id") or f"Chronos2-ETT-{self.frequency}-LoRA"
        if self.interval != FREQ_MINUTES[self.frequency]:
            raise ValueError("Manifest frequency and sampling interval do not agree")

    def predict_context(self, csv_path: str | Path, *, equipment_id: str,
                        operating_condition: str = "dataset_replay", episode_id: str | None = None) -> dict[str, Any]:
        series = load_series(Path(csv_path), Path(csv_path).stem, self.interval)
        values = series["values"]
        times = series["timestamps"]
        if len(values) < self.input_len:
            raise ValueError(f"At least {self.input_len} history rows required; got {len(values)}")
        history = values[-self.input_len:]
        history_times = times[-self.input_len:]
        item = {
            "target": history[:, -1].astype(np.float32),
            "past_covariates": {
                name: history[:, idx].astype(np.float32)
                for idx, name in enumerate(COLS[1:-1])
            },
        }
        quantiles, median = self.pipeline.predict_quantiles(
            inputs=[item], prediction_length=self.pred_len,
            quantile_levels=self.quantile_levels, context_length=self.input_len,
            batch_size=1, cross_learning=False)
        q = quantiles[0].detach().cpu().numpy() if hasattr(quantiles[0], "detach") else np.asarray(quantiles[0])
        med = median[0].detach().cpu().numpy() if hasattr(median[0], "detach") else np.asarray(median[0])
        q = q[0]  # [horizon, quantile]
        forecast = med[0] if med.ndim == 2 else med.reshape(-1)
        if len(forecast) != self.pred_len or not np.isfinite(forecast).all():
            raise ValueError("Chronos-2 returned an invalid forecast vector")
        quantile_vectors = {str(level): q[:, idx].astype(float).tolist()
                            for idx, level in enumerate(self.quantile_levels)}
        prediction_time = times[-1] + timedelta(minutes=self.interval)
        forecast_times = [(prediction_time + timedelta(minutes=i*self.interval)).isoformat(timespec="seconds")
                          for i in range(self.pred_len)]
        return {
            "episode_id": episode_id or f"CHRONOS2-ETT-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "prediction_time": prediction_time.isoformat(timespec="seconds"),
            "equipment_id": equipment_id, "operating_condition": operating_condition,
            "task_type": "transformer_oil_temperature_forecasting",
            "sensor_readings": {name: history[:, idx].astype(float).tolist()
                                for idx, name in enumerate(COLS[1:])},
            "model": {"model_name": "Chronos-2", "model_version": self.model_id,
                      "activation_id": 0, "checkpoint_sha256": self.manifest["model_artifact_sha256_tree"],
                      "model_task": "oil_temperature_regression", "frequency": self.frequency,
                      "base_model_id": self.manifest.get("base_model_id")},
            "model_prediction": {
                "prediction_task": "transformer_oil_temperature_forecasting",
                "prediction_kind": "probabilistic_regression", "target_name": "OT", "target_unit": "degC",
                "forecast_horizon": self.pred_len, "sample_interval_minutes": self.interval,
                "forecast_values": forecast.astype(float).tolist(), "forecast_times": forecast_times,
                "forecast_quantile_levels": self.quantile_levels,
                "forecast_quantiles": quantile_vectors,
                "forecast_p10": quantile_vectors.get("0.1"),
                "forecast_p50": quantile_vectors.get("0.5"),
                "forecast_p90": quantile_vectors.get("0.9"),
                "forecast_summary": {"first": float(forecast[0]), "last": float(forecast[-1]),
                                     "min": float(np.min(forecast)), "max": float(np.max(forecast)),
                                     "mean": float(np.mean(forecast))},
                # Quantile intervals are predictive intervals, not fault confidence or OOD calibration.
                "fault_type": None, "confidence": None, "confidence_calibrated": False,
                "ood_score": None, "drift_score": None, "input_quality": "valid"},
            "input_quality": "valid", "sensor_conflict": False,
            "source_data": {"path_name": Path(csv_path).name, "source_sha256": series["sha256"],
                            "history_rows": self.input_len,
                            "history_start": history_times[0].isoformat(),
                            "history_end": history_times[-1].isoformat(),
                            "dataset_sources": self.manifest.get("data_sources", []),
                            "sampling_interval_minutes": self.interval}}

    def run(self, agent: Any, csv_path: str | Path, **kwargs: Any) -> dict[str, Any]:
        context = self.predict_context(csv_path, **kwargs)
        report = agent.run(context)
        report["model_input_provenance"] = {
            "source": "Chronos2OilTemperatureBridge", "model_dir": str(self.model_dir),
            "model_artifact_sha256": self.manifest["model_artifact_sha256_tree"],
            "source_data": context["source_data"]}
        return report
