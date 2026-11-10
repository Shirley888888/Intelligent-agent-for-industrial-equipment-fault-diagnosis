"""Single Agent entry point for ETTh1 forecasting and industrial fault diagnosis."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .industrial_bridge import VersionedIndustrialFaultBridge
from .model_bridge import VersionedForecastModelBridge
from .ett_corpus_bridge import ETTOilTemperatureBridge


class TaskAwareModelRouter:
    """Route task-specific model outputs into the same DiagnosticAgent/EpisodeStore.

    Multi-task bundles execute each model task independently but persist all Episodes
    through the same Agent instance. Forecast values and fault scores are never fused.
    """

    FORECAST = "etth1_forecast"
    INDUSTRIAL = "industrial_fault_diagnosis"
    OIL_TEMPERATURE = "transformer_oil_temperature_forecasting"
    BUNDLE = "multi_task_bundle"

    def __init__(self, *, device: str = "cpu",
                 runtime_signal_provider: Callable[[np.ndarray, dict[str, Any]], dict[str, Any]] | None = None):
        self.device = device
        self.runtime_signal_provider = runtime_signal_provider

    def run(self, agent: Any, task_config: dict[str, Any]) -> dict[str, Any]:
        task_type = task_config.get("task_type")
        if task_type == self.BUNDLE:
            tasks = task_config.get("tasks")
            if not isinstance(tasks, list) or not tasks:
                raise ValueError("multi_task_bundle requires a nonempty tasks list")
            results = [self._run_one(agent, item) for item in tasks]
            return {
                "task_type": self.BUNDLE,
                "task_count": len(results),
                "episode_ids": [x.get("episode_id") for x in results],
                "results": results,
                "note": "Task outputs remain task-specific; no forecast/fault-score fusion is performed.",
            }
        return self._run_one(agent, task_config)

    def _run_one(self, agent: Any, task_config: dict[str, Any]) -> dict[str, Any]:
        task_type = task_config.get("task_type")
        if task_type == self.FORECAST:
            bridge = VersionedForecastModelBridge(task_config["model_root"], device=self.device)
            return bridge.run(agent, task_config["csv"],
                              equipment_id=task_config["equipment_id"],
                              operating_condition=task_config.get("operating_condition", "dataset_replay"),
                              episode_id=task_config.get("episode_id"))
        if task_type == self.OIL_TEMPERATURE:
            bridge = ETTOilTemperatureBridge(task_config["model_dir"], device=self.device)
            return bridge.run(agent, task_config["csv"],
                              equipment_id=task_config["equipment_id"],
                              operating_condition=task_config.get("operating_condition", "dataset_replay"),
                              episode_id=task_config.get("episode_id"))
        if task_type == self.INDUSTRIAL:
            bridge = VersionedIndustrialFaultBridge(
                task_config["model_root"], device=self.device,
                runtime_signal_provider=self.runtime_signal_provider)
            input_path = Path(task_config["input_json"])
            data = json.loads(input_path.read_text(encoding="utf-8"))
            history = np.asarray(data["history"], dtype=np.float32)
            return bridge.run(
                agent, history,
                equipment_id=data["equipment_id"],
                prediction_time=data["prediction_time"],
                operating_condition=data.get("operating_condition", "unspecified"),
                feature_names=data.get("feature_names"),
                sensor_readings=data.get("sensor_readings", {}),
                episode_id=data.get("episode_id"),
            )
        raise ValueError(f"Unsupported task_type: {task_type!r}; expected {self.FORECAST!r}, {self.OIL_TEMPERATURE!r}, {self.INDUSTRIAL!r} or {self.BUNDLE!r}")
