"""modify:tool-contracts - export schemas directly from the implemented signatures."""
from __future__ import annotations

import inspect
import types
from dataclasses import is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints, is_typeddict

import numpy as np
import minimal_loop as loop

TIME_SCHEMA = {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$",
               "description": "ISO 8601 dataset-local time; no UTC offset is asserted"}


def schema(annotation: Any) -> dict:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, types.UnionType):
        return {"anyOf": [schema(a) for a in args]}
    if origin is Literal:
        return {"enum": list(args)}
    if is_typeddict(annotation) or is_dataclass(annotation):
        hints = get_type_hints(annotation)
        properties = {k: schema(v) for k, v in hints.items()}
        if annotation is loop.CandidateResult:
            properties["train_config"] = schema(loop.TrainConfig)
            properties["manifest"] = schema(loop.DataVersion)
        if annotation is loop.GateResult:
            properties["gate_config"] = schema(loop.GateConfig)
        if annotation is loop.FeedbackEpisode:
            properties["history"].update(minItems=loop.L, maxItems=loop.L)
            properties["history"]["items"].update(minItems=7, maxItems=7)
            for key, length in [("history_times", loop.L), ("target_times", loop.H), ("forecast", loop.H)]:
                properties[key].update(minItems=length, maxItems=length)
                if key.endswith("times"):
                    properties[key]["items"] = dict(TIME_SCHEMA)
            properties["truth"]["anyOf"][0].update(minItems=loop.H, maxItems=loop.H)
            properties["prediction_time"] = dict(TIME_SCHEMA)
        if "as_of" in properties:
            properties["as_of"] = dict(TIME_SCHEMA)
        return {"type": "object", "title": annotation.__name__,
                "properties": properties,
                "required": list(hints), "additionalProperties": False}
    if origin is list:
        return {"type": "array", "items": schema(args[0])}
    if origin is tuple:
        return {"type": "array", "prefixItems": [schema(a) for a in args],
                "minItems": len(args), "maxItems": len(args)}
    if annotation is list:
        return {"type": "array"}
    if origin is dict:
        return {"type": "object", "additionalProperties": schema(args[1])}
    if annotation in (dict, Any):
        return {"type": "object"} if annotation is dict else {}
    if annotation is np.ndarray:
        return {"type": "array", "items": {"type": "array", "items": {"type": "number"},
                                             "minItems": 7, "maxItems": 7},
                "x-python-type": "numpy.ndarray", "description": "Observed values [T,7], finite"}
    if annotation is datetime:
        return {**TIME_SCHEMA, "x-python-type": "datetime"}
    if annotation is Path:
        return {"type": "string", "x-python-type": "pathlib.Path"}
    if annotation is loop.ForecastModel:
        return {"type": "object", "x-python-type": "ForecastModel",
                "description": "In-process model handle loaded from a checkpoint; not a JSON array"}
    basic = {str: "string", int: "integer", float: "number", bool: "boolean", type(None): "null"}
    return {"type": basic[annotation]} if annotation in basic else {"x-python-type": str(annotation)}


def contract(name: str, function: Any, effect: str, constraints: list[str]) -> dict:
    signature, hints = inspect.signature(function), get_type_hints(function)
    properties, required = {}, []
    for key, parameter in signature.parameters.items():
        if key == "self":
            continue
        item = schema(hints.get(key, Any))
        if parameter.default is inspect.Parameter.empty:
            required.append(key)
        else:
            item["default"] = parameter.default
        properties[key] = item
    return {"name": name, "entrypoint": function.__qualname__, "signature": str(signature),
            "input_schema": {"type": "object", "properties": properties,
                             "required": required, "additionalProperties": False},
            "output_schema": schema(hints.get("return", Any)),
            "side_effect": effect, "constraints": constraints,
            "errors": ["ValueError: invalid values, shapes, timestamps, splits or stale gate",
                       "FileNotFoundError/OSError: unavailable local artifact",
                       "RuntimeError/ImportError: unavailable PyTorch/CUDA backend"]}


def tool_catalog() -> dict:
    entries = [
        ("Prediction", loop.predict_tool, "none; caller appends predictions.jsonl",
         ["history [96,7]; exactly 96 hourly timestamps", "all history times < prediction_time",
          "forecast [24]; truth/MAE initially null"]),
        ("Feedback Collector", loop.complete_feedback, "none; returns a new completed Episode",
         ["input is only observed prefix [0,T)", "truth times align exactly with 24 target times",
          "last target time < as_of", "original forecast and model_version preserved"]),
        ("Feedback Monitor", loop.FeedbackMonitor.run, "updates local checked/streak/cooldown state",
         ["current model_version + activation_id only", "latest N complete episodes for rolling MAE",
          "triggers require consecutive nonoverlapping degraded blocks and cooldown"]),
        ("Train Candidate", loop.train_candidate_tool, "writes a new candidate directory",
         ["only train_ranges fit weights/scaler", "validation selects epoch; gate targets excluded",
          "one fixed configuration per candidate; existing model never overwritten"]),
        ("Model Gate", loop.model_gate_tool, "none; caller writes gate.json",
         ["same recent and historical windows for both models", "each model uses its own scaler",
          "both quantitative conditions must pass", "activation ID/hash/snapshot bound for publication"]),
        ("Publish Version", loop.VersionManager.publish, "immutable M directory + atomic current.json",
         ["M0 bootstrap permits no gate", "later publication requires current activation/hash binding",
          "version numbers never reused after rollback", "failed activation removes only the new unactivated bundle"]),
        ("Rollback", loop.VersionManager.activate, "event append followed by atomic current.json commit",
         ["load/checksum/shape/scaler/inference verified before switch", "clock cannot move backwards",
          "activation_id increments; historical predictions unchanged",
          "single writer; precommit I/O exceptions restore event bytes; no crash-atomicity guarantee"]),
        ("Load Version", loop.VersionManager.load, "none", ["returns model handle and saved manifest"]),
        ("Current Version", loop.VersionManager.current, "none", ["returns current pointer"]),
        ("List Versions", loop.VersionManager.list_versions, "none", ["checks each historical bundle"]),
    ]
    return {"schema_version": 2, "scope": "Tools implemented in this uploaded standalone loop",
            "excluded_modules": ["Risk", "Anomaly", "RAG", "Diagnosis: not present in this archive"],
            "features": loop.FEATURES, "input_hours": loop.L, "forecast_hours": loop.H,
            "target": "OT", "metric_unit": "degC", "timestamp_policy": "ETTh1 naive dataset-local hourly time",
            "constructors": {"FeedbackMonitor": str(inspect.signature(loop.FeedbackMonitor)),
                             "VersionManager": str(inspect.signature(loop.VersionManager))},
            "structures": {t.__name__: schema(t) for t in
                           (loop.FeedbackEpisode, loop.DataVersion, loop.TrainConfig, loop.GateConfig,
                            loop.CandidateResult, loop.MonitorDecision, loop.GateResult, loop.VersionPointer)},
            "tools": [contract(*entry) for entry in entries]}
