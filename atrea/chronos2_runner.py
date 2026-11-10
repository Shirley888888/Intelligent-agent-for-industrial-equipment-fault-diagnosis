"""Adapter that connects ATREA to the repository's real Chronos-2 oil-temperature trainer.

This adapter performs an actual Chronos-2 fine-tuning run when explicitly invoked.
It does not alter the pretrained architecture; it allows controlled training-config
search (context, steps, batch, learning rate, LoRA/full) and returns comparable metrics.
"""
from __future__ import annotations
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

_ALLOWED = {
    "data_dir", "frequency", "datasets", "eval_datasets", "base_model", "revision",
    "device_map", "out_dir", "prediction_length", "context_length", "num_steps", "batch_size",
    "gradient_accumulation_steps", "learning_rate", "finetune_mode", "test_stride", "seed",
}
_PLAN_ALLOWED = {
    "frequency", "datasets", "eval_datasets", "prediction_length", "context_length",
    "num_steps", "batch_size", "gradient_accumulation_steps", "learning_rate",
    "finetune_mode", "test_stride", "seed",
}


def _aggregate_metrics(summary):
    agg = summary.get("aggregate", {})
    mapping = {
        "mae": "MAE_degC", "rmse": "RMSE_degC", "p99_ae": "P99_AE_degC",
        "max_ae": "MaxAE_degC", "interval_coverage": "P10_P90_coverage",
        "interval_width": "P10_P90_mean_width_degC",
    }
    out = {}
    for target, source in mapping.items():
        value = agg.get(source)
        if isinstance(value, (int, float)):
            out[target] = float(value)
    if "mae" not in out:
        raise ValueError("Chronos-2 manifest is missing aggregate MAE_degC")
    return out


class Chronos2ExperimentRunner:
    """Callable ATREA runner: runner(proposal, batch_id) -> baseline/candidate records."""
    def __init__(self, repo_root, *, base_config=None, python_executable=None, timeout_seconds=None):
        self.repo_root = Path(repo_root).resolve()
        self.script = self.repo_root / "train_chronos2_ett.py"
        if not self.script.is_file():
            raise FileNotFoundError(f"Chronos-2 trainer not found: {self.script}")
        self.base_config = dict(base_config or {})
        unknown = set(self.base_config) - _ALLOWED
        if unknown:
            raise ValueError(f"Unsupported Chronos-2 base config keys: {sorted(unknown)}")
        self.python = python_executable or sys.executable
        self.timeout_seconds = timeout_seconds

    def __call__(self, proposal, batch_id):
        plan = proposal.get("plan", {}) if isinstance(proposal, dict) else {}
        overrides = plan.get("chronos2_config", {})
        if not isinstance(overrides, dict):
            raise TypeError("plan.chronos2_config must be a dict")
        unknown = set(overrides) - _PLAN_ALLOWED
        if unknown:
            raise ValueError(f"Unsupported or privileged Chronos-2 plan overrides: {sorted(unknown)}")
        config = {**self.base_config, **overrides}
        # Separate output directory prevents accidental overwrite and preserves provenance.
        run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(batch_id))[:80]
        config["run_name"] = run_name
        config.setdefault("out_dir", str(self.repo_root / "models" / "atrea_chronos2_runs"))
        config.setdefault("data_dir", str(self.repo_root / "data" / "ett"))
        for path_key in ("out_dir", "data_dir"):
            path_value = Path(config[path_key])
            if not path_value.is_absolute():
                path_value = self.repo_root / path_value
            config[path_key] = str(path_value.resolve())
        config_path = self.repo_root / "agent_workspace" / "chronos2_configs" / f"{run_name}.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        command = [self.python, str(self.script), "--config", str(config_path)]
        completed = subprocess.run(command, cwd=self.repo_root, text=True,
            capture_output=True, timeout=self.timeout_seconds, check=False)
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout)[-12000:]
            raise RuntimeError(f"Chronos-2 training failed (exit={completed.returncode}):\n{tail}")
        manifest_path = Path(config["out_dir"]) / run_name / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Training finished but manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        baseline = _aggregate_metrics(manifest["zero_shot_test_metrics"])
        candidate = _aggregate_metrics(manifest["candidate_test_metrics"])
        split_payload = {
            "frequency": manifest.get("frequency"),
            "prediction_length": manifest.get("prediction_length"),
            "test_stride": config.get("test_stride"),
            "training_sources": manifest.get("training_data_sources"),
            "evaluation_sources": manifest.get("evaluation_data_sources"),
            "split_policy": manifest.get("split_policy"),
        }
        split_hash = hashlib.sha256(json.dumps(split_payload, sort_keys=True,
            ensure_ascii=False).encode("utf-8")).hexdigest()
        data_hash = hashlib.sha256(json.dumps(manifest.get("data_sources", []), sort_keys=True,
            ensure_ascii=False).encode("utf-8")).hexdigest()
        common = {
            "dataset": "+".join(x.get("dataset", "unknown") for x in manifest.get("evaluation_data_sources", [])),
            "split_hash": split_hash, "data_hash": data_hash,
            "code_version": hashlib.sha256(self.script.read_bytes()).hexdigest(),
            "config": config, "hyperparameters": {k: config.get(k) for k in
                ("learning_rate", "num_steps", "batch_size", "gradient_accumulation_steps", "context_length", "prediction_length", "finetune_mode")},
            "seed": config.get("seed"), "model_version": manifest.get("base_model_commit"),
            "artifact_dir": str(manifest_path.parent),
        }
        return [
            {**common, "model_name":"Chronos-2-zero-shot", "metrics":baseline,
             "status":"completed", "is_candidate":False, "architecture_changed":False},
            {**common, "model_name":"Chronos-2-finetuned", "metrics":candidate,
             "status":"completed", "is_candidate":True, "architecture_changed":False,
             "checkpoint_path":str(manifest_path.parent / manifest.get("model_artifact", "finetuned-ckpt")),
             "candidate_gate_validation":manifest.get("candidate_gate", {}),
             "manifest_path":str(manifest_path)},
        ]


def build_runner_from_env():
    """Factory for the scheduler: configure ATREA_REPO_ROOT and optional ATREA_CHRONOS2_CONFIG."""
    import os
    repo_root = os.getenv("ATREA_REPO_ROOT", str(Path(__file__).resolve().parents[1]))
    config_path = os.getenv("ATREA_CHRONOS2_CONFIG")
    config = json.loads(Path(config_path).read_text(encoding="utf-8")) if config_path else {}
    timeout = os.getenv("ATREA_TRAIN_TIMEOUT_SECONDS")
    return Chronos2ExperimentRunner(repo_root, base_config=config,
        timeout_seconds=int(timeout) if timeout else None)
