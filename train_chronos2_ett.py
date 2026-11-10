#!/usr/bin/env python3
"""Fine-tune the Chronos-2 time-series foundation model on real ETT oil-temperature data.

ETTh1/ETTh2 (hourly) and ETTm1/ETTm2 (15-minute) are trained separately.
The target is OT (transformer oil temperature); the six load channels are past-only
covariates because future load values are not assumed known at inference time.

This script requires optional Chronos/Hugging Face dependencies and internet access
for the first download of amazon/chronos-2. The current offline build environment
cannot download those weights; the script is packaged for execution on the user's
networked GPU workstation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from download_ett import SOURCE_COMMIT, git_blob_sha1, validate_csv
from train_ett_corpus import (
    COLS, DATASETS, FREQ_MINUTES, FILES_FREQ, load_series,
)

BASE_MODEL = "amazon/chronos-2"
LOAD_FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
QUANTILE_LEVELS = [0.1, 0.5, 0.9]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_tree(root: Path) -> str:
    """Stable digest over relative file names and file contents."""
    h = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    for path in files:
        rel = path.relative_to(root).as_posix().encode("utf-8")
        h.update(len(rel).to_bytes(8, "big")); h.update(rel)
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


def chronological_partitions(n: int) -> tuple[int, int, int]:
    """Return 60% train, 70% fine-tune validation, 80% gate-validation cutoffs.

    The [70%, 80%) interval is reserved for candidate-vs-zero-shot gate checks and
    is not passed into Chronos2Pipeline.fit(). The final 20% stays untouched for
    final reporting. This prevents using the same validation target windows both
    for HF early stopping and candidate promotion decisions.
    """
    if n < 1000:
        raise ValueError(f"ETT series is unexpectedly short: {n} rows")
    train_end = int(n * 0.60)
    fit_val_end = int(n * 0.70)
    gate_end = int(n * 0.80)
    if (train_end <= 96 or fit_val_end - train_end < 48
            or gate_end - fit_val_end < 48 or n - gate_end < 24):
        raise ValueError("Series is too short for train/fit-validation/gate/test windows")
    return train_end, fit_val_end, gate_end


def chronological_bounds(n: int) -> tuple[int, int]:
    """Backward-compatible train and final-test boundary helper."""
    train_end, _, gate_end = chronological_partitions(n)
    return train_end, gate_end


def chronos_input(values: np.ndarray, end: int) -> dict[str, Any]:
    """Build a Chronos-2 raw input with OT target and six *past-only* load covariates."""
    if values.ndim != 2 or values.shape[1] != len(COLS) - 1:
        raise ValueError(f"values must have shape (time, 7), got {values.shape}")
    if not 1 <= end <= len(values):
        raise ValueError("end index outside series")
    part = values[:end]
    if not np.isfinite(part).all():
        raise ValueError("non-finite value in Chronos input")
    # In the ETT feature order OT is the final column; all load features are past-only.
    return {
        "target": part[:, -1].astype(np.float32, copy=True),
        "past_covariates": {
            name: part[:, idx].astype(np.float32, copy=True)
            for idx, name in enumerate(LOAD_FEATURES)
        },
    }


def make_fit_inputs(series_list: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Build train and HF early-stop validation inputs without consuming gate/test tails."""
    train_inputs, validation_inputs, split = [], [], {}
    for series in series_list:
        n = len(series["values"])
        train_end, fit_val_end, gate_end = chronological_partitions(n)
        train_inputs.append(chronos_input(series["values"], train_end))
        # The final prediction_length observations inside this 70% prefix are used
        # by Chronos2's DatasetMode.VALIDATION for its own early-stopping evaluation.
        validation_inputs.append(chronos_input(series["values"], fit_val_end))
        split[series["name"]] = {
            "train_end_exclusive": train_end,
            "fit_validation_end_exclusive": fit_val_end,
            "gate_validation_start_inclusive": fit_val_end,
            "gate_validation_end_exclusive": gate_end,
            "test_start_inclusive": gate_end,
            "test_end_exclusive": n,
        }
    if not train_inputs:
        raise ValueError("No training series provided")
    # Chronos2 requires consistent covariate schema across a list of dict inputs.
    schemas = [tuple(sorted(x["past_covariates"].keys())) for x in train_inputs + validation_inputs]
    if len(set(schemas)) != 1:
        raise ValueError("All Chronos inputs must have identical covariate schemas")
    return train_inputs, validation_inputs, split


def regression_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    y = np.asarray(truth, dtype=np.float64).reshape(-1)
    if p.shape != y.shape or not len(y) or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("metrics require equal, non-empty finite prediction and truth arrays")
    ae = np.abs(p - y)
    return {"MAE_degC": float(ae.mean()),
            "RMSE_degC": float(np.sqrt(np.mean((p-y)**2))),
            "P95_AE_degC": float(np.quantile(ae, 0.95)),
            "P99_AE_degC": float(np.quantile(ae, 0.99)),
            "MaxAE_degC": float(ae.max())}


def evaluate_rolling(pipeline: Any, series_list: list[dict[str, Any]],
                     pred_len: int, test_stride: int, context_length: int,
                     batch_size: int = 16, segment: str = "test") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate rolling-origin forecasts on gate-validation [70%,80%) or test [80%,100%)."""
    if segment not in {"gate", "test"}:
        raise ValueError("segment must be 'gate' or 'test'")
    all_inputs: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    split_rows: list[tuple[str, int, np.ndarray]] = []
    for series in series_list:
        values = series["values"]
        _, fit_val_end, gate_end = chronological_partitions(len(values))
        start, end = (fit_val_end, gate_end) if segment == "gate" else (gate_end, len(values))
        for origin in range(start, end - pred_len + 1, test_stride):
            all_inputs.append(chronos_input(values, origin))
            split_rows.append((series["name"], origin,
                               values[origin:origin+pred_len, -1].astype(np.float32, copy=True)))
    if not all_inputs:
        raise ValueError("No test windows available")

    by_series: dict[str, dict[str, list[np.ndarray]]] = {
        s["name"]: {"pred": [], "truth": [], "lower": [], "upper": []}
        for s in series_list
    }
    for start in range(0, len(all_inputs), batch_size):
        batch = all_inputs[start:start+batch_size]
        quantiles, median = pipeline.predict_quantiles(
            inputs=batch, prediction_length=pred_len,
            quantile_levels=QUANTILE_LEVELS, batch_size=batch_size,
            context_length=context_length, cross_learning=False)
        for local_i, (name, origin, truth) in enumerate(split_rows[start:start+batch_size]):
            q = quantiles[local_i]
            # Chronos-2 returns [n_targets, horizon, n_quantiles]. OT is the sole target.
            q_np = q.detach().cpu().numpy() if hasattr(q, "detach") else np.asarray(q)
            med_np = median[local_i].detach().cpu().numpy() if hasattr(median[local_i], "detach") else np.asarray(median[local_i])
            pred = med_np[0] if med_np.ndim == 2 else med_np.reshape(-1)
            lower, upper = q_np[0, :, 0], q_np[0, :, 2]
            by_series[name]["pred"].append(np.asarray(pred, dtype=np.float32))
            by_series[name]["truth"].append(truth)
            by_series[name]["lower"].append(np.asarray(lower, dtype=np.float32))
            by_series[name]["upper"].append(np.asarray(upper, dtype=np.float32))
            records.append({"series": name, "origin_index": origin,
                            "prediction": np.asarray(pred).tolist(),
                            "p10": np.asarray(lower).tolist(), "p90": np.asarray(upper).tolist(),
                            "truth": truth.tolist()})

    per_series: dict[str, Any] = {}
    aggregate_pred, aggregate_truth, aggregate_lower, aggregate_upper = [], [], [], []
    for name, parts in by_series.items():
        p = np.concatenate(parts["pred"]); y = np.concatenate(parts["truth"])
        lo = np.concatenate(parts["lower"]); hi = np.concatenate(parts["upper"])
        per_series[name] = {**regression_metrics(p, y),
                            "P10_P90_coverage": float(np.mean((y >= lo) & (y <= hi))),
                            "P10_P90_mean_width_degC": float(np.mean(hi-lo)),
                            "test_windows": int(len(parts["pred"]))}
        aggregate_pred.append(p); aggregate_truth.append(y)
        aggregate_lower.append(lo); aggregate_upper.append(hi)
    ap, ay = np.concatenate(aggregate_pred), np.concatenate(aggregate_truth)
    alo, ahi = np.concatenate(aggregate_lower), np.concatenate(aggregate_upper)
    summary = {"aggregate": {**regression_metrics(ap, ay),
                             "P10_P90_coverage": float(np.mean((ay >= alo) & (ay <= ahi))),
                             "P10_P90_mean_width_degC": float(np.mean(ahi-alo)),
                             "test_windows": len(records)},
               "per_series": per_series}
    return summary, records


def candidate_gate(baseline: dict[str, Any], candidate: dict[str, Any],
                   overall_mae_allow_pct: float = 1.0,
                   station_mae_allow_pct: float = 3.0,
                   p99_allow_pct: float = 5.0,
                   maxae_allow_pct: float = 10.0,
                   min_interval_coverage: float = 0.70,
                   interval_width_allow_pct: float = 20.0) -> dict[str, Any]:
    """Conservative validation gate; it never auto-promotes a candidate.

    P10-P90 is nominally an 80% interval. A 70% minimum coverage is an initial
    under-coverage guard; interval width is also bounded to discourage trivially
    wide intervals that achieve coverage without useful precision.
    """
    b = baseline["aggregate"]; c = candidate["aggregate"]
    overall_regression = (c["MAE_degC"] / max(b["MAE_degC"], 1e-12) - 1.0) * 100
    p99_regression = (c["P99_AE_degC"] / max(b["P99_AE_degC"], 1e-12) - 1.0) * 100
    maxae_regression = (c["MaxAE_degC"] / max(b["MaxAE_degC"], 1e-12) - 1.0) * 100
    width_regression = (c["P10_P90_mean_width_degC"] /
                        max(b["P10_P90_mean_width_degC"], 1e-12) - 1.0) * 100
    station_mae: dict[str, float] = {}
    station_coverage: dict[str, float] = {}
    for name, base_metrics in baseline["per_series"].items():
        cand_metrics = candidate["per_series"][name]
        station_mae[name] = (cand_metrics["MAE_degC"] /
                             max(base_metrics["MAE_degC"], 1e-12) - 1.0) * 100
        station_coverage[name] = float(cand_metrics["P10_P90_coverage"])
    candidate_coverage = float(c["P10_P90_coverage"])
    checks = {
        "aggregate_mae_non_regression": overall_regression <= overall_mae_allow_pct,
        "aggregate_p99_non_regression": p99_regression <= p99_allow_pct,
        "aggregate_maxae_non_regression": maxae_regression <= maxae_allow_pct,
        "each_station_mae_non_regression": all(v <= station_mae_allow_pct for v in station_mae.values()),
        "aggregate_interval_coverage_floor": candidate_coverage >= min_interval_coverage,
        "each_station_interval_coverage_floor": all(v >= min_interval_coverage for v in station_coverage.values()),
        "interval_width_not_excessively_wider": width_regression <= interval_width_allow_pct,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "aggregate_mae_regression_pct": overall_regression,
            "aggregate_p99_regression_pct": p99_regression,
            "aggregate_maxae_regression_pct": maxae_regression,
            "aggregate_interval_width_regression_pct": width_regression,
            "candidate_interval_coverage": candidate_coverage,
            "station_mae_regression_pct": station_mae,
            "station_interval_coverage": station_coverage,
            "thresholds": {"overall_mae_allow_pct": overall_mae_allow_pct,
                           "station_mae_allow_pct": station_mae_allow_pct,
                           "p99_allow_pct": p99_allow_pct,
                           "maxae_allow_pct": maxae_allow_pct,
                           "min_interval_coverage": min_interval_coverage,
                           "interval_width_allow_pct": interval_width_allow_pct},
            "note": "Gate is evaluated on a reserved 70%-80% validation segment, not on the final 20% test set. Passing still requires independent test review, project model gate, shadow replay and joint release review."}


def require_finetune_dependencies(finetune_mode: str) -> None:
    """Fail closed when LoRA was requested but PEFT is unavailable.

    Chronos-2 may otherwise warn and silently fall back to full fine-tuning,
    which would make the run configuration and manifest misleading.
    """
    if finetune_mode == "lora":
        try:
            import peft  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "LoRA fine-tuning was requested, but PEFT is not installed. "
                "Install the pinned Chronos dependencies from requirements_chronos.txt; "
                "refusing to silently fall back to full fine-tuning."
            ) from exc
    elif finetune_mode != "full":
        raise ValueError(f"Unsupported finetune_mode: {finetune_mode!r}; expected 'lora' or 'full'")


def _load_chronos_pipeline(model_id: str, revision: str | None, device_map: str):
    try:
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Chronos-2 dependencies are missing. Install with: pip install -r requirements_chronos.txt"
        ) from exc
    kwargs = {"device_map": device_map}
    if revision:
        kwargs["revision"] = revision
    return Chronos2Pipeline.from_pretrained(model_id, **kwargs)


def train_frequency(series_list: list[dict[str, Any]], frequency: str, *,
                    output_dir: Path, base_model: str, revision: str | None,
                    device_map: str, prediction_length: int, context_length: int,
                    num_steps: int, batch_size: int, learning_rate: float,
                    finetune_mode: str, seed: int, test_stride: int,
                    gradient_accumulation_steps: int = 4,
                    evaluation_series_list: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    training_series_list = series_list
    evaluation_series_list = evaluation_series_list or series_list
    if not training_series_list or not evaluation_series_list:
        raise ValueError("Both training and evaluation series lists must be non-empty")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_inputs, val_inputs, train_split = make_fit_inputs(training_series_list)
    _, _, evaluation_split = make_fit_inputs(evaluation_series_list)
    train_names = {s["name"] for s in training_series_list}
    eval_names = {s["name"] for s in evaluation_series_list}
    overlapping_names = sorted(train_names.intersection(eval_names))
    print(f"[DATA] train_series={sorted(train_names)} eval_series={sorted(eval_names)} "
          f"overlap={overlapping_names}", flush=True)
    require_finetune_dependencies(finetune_mode)
    print(f"[LOAD] {base_model} device_map={device_map}", flush=True)
    base_pipeline = _load_chronos_pipeline(base_model, revision, device_map)
    base_commit = getattr(getattr(base_pipeline, "model", None), "config", None)
    base_commit = getattr(base_commit, "_commit_hash", None)

    print("[EVAL] zero-shot foundation model on reserved gate-validation 70%-80%", flush=True)
    baseline_gate_metrics, baseline_gate_records = evaluate_rolling(
        base_pipeline, evaluation_series_list, prediction_length, test_stride, context_length, batch_size, segment="gate")
    print("[EVAL] zero-shot foundation model on untouched final test 80%-100%", flush=True)
    baseline_test_metrics, baseline_test_records = evaluate_rolling(
        base_pipeline, evaluation_series_list, prediction_length, test_stride, context_length, batch_size, segment="test")
    (output_dir / "zero_shot_metrics.json").write_text(
        json.dumps(baseline_test_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "gate_validation_zero_shot_metrics.json").write_text(
        json.dumps(baseline_gate_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[FIT] mode={finetune_mode} steps={num_steps} context={context_length}", flush=True)
    # Chronos2Pipeline.fit clones the pretrained model before optimizing; the zero-shot base remains unchanged.
    finetuned = base_pipeline.fit(
        inputs=train_inputs, validation_inputs=val_inputs,
        prediction_length=prediction_length, finetune_mode=finetune_mode,
        context_length=context_length, learning_rate=learning_rate,
        num_steps=num_steps, batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        output_dir=output_dir, finetuned_ckpt_name="finetuned-ckpt",
        min_past=prediction_length, remove_printer_callback=True,
        seed=seed, data_seed=seed,
    )

    print("[EVAL] fine-tuned candidate on reserved gate-validation 70%-80%", flush=True)
    candidate_gate_metrics, candidate_gate_records = evaluate_rolling(
        finetuned, evaluation_series_list, prediction_length, test_stride, context_length, batch_size, segment="gate")
    gate = candidate_gate(baseline_gate_metrics, candidate_gate_metrics)
    print("[EVAL] fine-tuned candidate on untouched final test 80%-100%", flush=True)
    candidate_test_metrics, candidate_test_records = evaluate_rolling(
        finetuned, evaluation_series_list, prediction_length, test_stride, context_length, batch_size, segment="test")
    model_dir = output_dir / "finetuned-ckpt"
    if not model_dir.exists():
        raise RuntimeError(f"Chronos-2 fit completed but checkpoint directory missing: {model_dir}")
    model_hash = sha256_tree(model_dir)
    def source_rows(items: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
        rows = []
        for item in items:
            raw_path = item.get("path")
            csv_path = Path(raw_path) if raw_path else None
            # Synthetic unit-test fixtures may not exist on disk; real training paths
            # are source-verified in load_named() before this manifest is written.
            blob_sha = git_blob_sha1(csv_path) if csv_path is not None and csv_path.is_file() else None
            rows.append({"dataset": item["name"], "role": role, "rows": item["rows"],
                         "sha256": item["sha256"], "github_blob_sha1": blob_sha,
                         "source_commit": SOURCE_COMMIT, "start": item["start"], "end": item["end"],
                         "source_url": f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/{SOURCE_COMMIT}/ETT-small/{item['name']}.csv"})
        return rows
    training_sources = source_rows(training_series_list, "training_and_hf_validation")
    evaluation_sources = source_rows(evaluation_series_list, "candidate_gate_and_final_test")
    manifest = {
        "schema_version": 1, "task": "transformer_oil_temperature_forecasting",
        "model_id": f"Chronos2-ETT-{frequency}-{finetune_mode.upper()}",
        "model_family": "Chronos-2", "parameter_count_claim": "120M (upstream model card; not locally re-counted)",
        "base_model_id": base_model, "base_model_revision_requested": revision or "default upstream revision",
        "base_model_commit": base_commit, "finetune_mode": finetune_mode,
        "model_artifact": "finetuned-ckpt", "model_artifact_sha256_tree": model_hash,
        "target": "OT", "target_unit": "degC",
        "past_covariates": LOAD_FEATURES, "future_load_assumed_known": False,
        "frequency": frequency, "sample_interval_minutes": FREQ_MINUTES[frequency],
        "prediction_length": prediction_length, "context_length": context_length,
        "batch_size_per_device": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size_per_device": batch_size * gradient_accumulation_steps,
        "num_steps": num_steps,
        "input_duration_hours": context_length * FREQ_MINUTES[frequency] / 60,
        "forecast_duration_hours": prediction_length * FREQ_MINUTES[frequency] / 60,
        "quantile_levels": QUANTILE_LEVELS,
        "training_data_sources": training_sources,
        "evaluation_data_sources": evaluation_sources,
        "data_sources": training_sources + evaluation_sources,
        "evaluation_design": ("same_station_temporal_holdout" if train_names == eval_names else
                              "cross_station_holdout" if train_names.isdisjoint(eval_names) else
                              "mixed_station_overlap_eval"),
        "split_policy": "chronological_per_series_60_10_10_20; train prefix 0-60%; Chronos fit-validation prefix 0-70% (last horizon is HF validation target); reserved gate-validation rolling targets 70-80%; final test rolling targets 80-100%",
        "training_split_ranges": train_split,
        "evaluation_split_ranges": evaluation_split,
        "zero_shot_gate_validation_metrics": baseline_gate_metrics,
        "candidate_gate_validation_metrics": candidate_gate_metrics,
        "zero_shot_test_metrics": baseline_test_metrics,
        "candidate_test_metrics": candidate_test_metrics,
        "candidate_gate": gate,
        "release_status": "gate_validation_passed_pending_independent_test_and_joint_release_review" if gate["passed"] else "candidate_rejected_by_gate_validation",
        "agent_activation": False,
        "dataset_license": "CC BY-ND 4.0; original files are not redistributed in this package",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (output_dir / "candidate_metrics.json").write_text(
        json.dumps(candidate_test_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "gate_validation_candidate_metrics.json").write_text(
        json.dumps(candidate_gate_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "candidate_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "zero_shot_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in baseline_test_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "candidate_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in candidate_test_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "gate_validation_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in candidate_gate_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"frequency": frequency,
                      "zero_shot_gate_MAE": baseline_gate_metrics["aggregate"]["MAE_degC"],
                      "candidate_gate_MAE": candidate_gate_metrics["aggregate"]["MAE_degC"],
                      "zero_shot_test_MAE": baseline_test_metrics["aggregate"]["MAE_degC"],
                      "candidate_test_MAE": candidate_test_metrics["aggregate"]["MAE_degC"],
                      "gate_passed": gate["passed"], "model_dir": str(model_dir)}, indent=2), flush=True)
    return manifest


def resolve_run_dir(out_root: Path, frequency: str, run_name: str | None = None,
                    all_frequencies: bool = False) -> Path:
    if run_name and all_frequencies:
        raise ValueError("--run-name cannot be used with --frequency all")
    safe_name = run_name or frequency
    if Path(safe_name).name != safe_name or safe_name in {".", ".."}:
        raise ValueError("run_name must be a single safe directory name")
    return out_root / safe_name


def main() -> None:
    # Load JSON defaults first, then let explicit CLI arguments override them.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    pre_args, _ = pre.parse_known_args()
    config_defaults: dict[str, Any] = {}
    if pre_args.config:
        cfg_path = Path(pre_args.config)
        config_defaults = json.loads(cfg_path.read_text(encoding="utf-8"))
        supported = {"data_dir", "frequency", "datasets", "eval_datasets", "base_model", "revision", "device_map",
                     "out_dir", "prediction_length", "context_length", "num_steps", "batch_size",
                     "gradient_accumulation_steps", "learning_rate", "finetune_mode", "test_stride", "seed", "run_name"}
        unknown = set(config_defaults) - supported - {"target", "past_covariates"}
        if unknown:
            raise ValueError(f"Unsupported Chronos config keys: {sorted(unknown)}")
        if config_defaults.get("target", "OT") != "OT":
            raise ValueError("This pipeline is specifically for OT (transformer oil temperature)")
        if config_defaults.get("past_covariates", LOAD_FEATURES) != LOAD_FEATURES:
            raise ValueError(f"past_covariates must be exactly {LOAD_FEATURES}")
        config_defaults = {k: v for k, v in config_defaults.items() if k not in {"target", "past_covariates"}}

    p = argparse.ArgumentParser(description="LoRA/full fine-tune Chronos-2 on real transformer oil-temperature data")
    p.add_argument("--config", default=None, help="Optional JSON run configuration; CLI flags override it")
    p.add_argument("--data-dir", default="data/ett")
    p.add_argument("--frequency", choices=["hourly", "15min", "all"], default="hourly")
    p.add_argument("--datasets", nargs="+", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
                   help="Series used for fine-tuning (training + HF validation)")
    p.add_argument("--eval-datasets", nargs="+", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
                   help="Series used for reserved gate and final test; use the other station for leave-one-station-out")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--revision", default=None)
    p.add_argument("--device-map", default="auto", help="Hugging Face device_map: auto, cuda, cpu")
    p.add_argument("--out-dir", default="models/chronos2_ett")
    p.add_argument("--run-name", default=None,
                   help="Optional specific run folder under --out-dir (useful for LOSO runs)")
    p.add_argument("--preflight-only", action="store_true",
                   help="Validate downloaded datasets and temporal splits without loading Chronos weights or training")
    p.add_argument("--prediction-length", type=int, default=24)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=4,
                   help="Per-device micro-batch; 4 is a conservative starting point for 8GB GPUs")
    p.add_argument("--gradient-accumulation-steps", type=int, default=4,
                   help="Accumulate gradients to approximate a larger effective batch")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--finetune-mode", choices=["lora", "full"], default="lora")
    p.add_argument("--test-stride", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(**config_defaults)
    args = p.parse_args()
    if args.prediction_length <= 0 or args.context_length < args.prediction_length:
        p.error("Require prediction-length > 0 and context-length >= prediction-length")
    if (args.num_steps <= 0 or args.batch_size <= 0 or args.test_stride <= 0
            or args.gradient_accumulation_steps <= 0):
        p.error("num-steps, batch-size, gradient-accumulation-steps and test-stride must be positive")
    data_dir, out_root = Path(args.data_dir), Path(args.out_dir)
    frequencies = ["hourly", "15min"] if args.frequency == "all" else [args.frequency]
    if args.datasets and len(frequencies) != 1:
        p.error("--datasets can only be used with one --frequency value")
    if args.run_name and len(frequencies) != 1:
        p.error("--run-name can only be used with one --frequency value")
    results = []
    for frequency in frequencies:
        names = args.datasets or DATASETS[frequency]
        expected_frequency = "hourly" if frequency == "hourly" else "15min"
        if any(FILES_FREQ(name) != expected_frequency for name in names):
            p.error(f"Datasets {names} do not belong to frequency={frequency}")
        eval_names = args.eval_datasets or names
        if any(FILES_FREQ(name) != expected_frequency for name in eval_names):
            p.error(f"Evaluation datasets {eval_names} do not belong to frequency={frequency}")
        def load_named(items: list[str]) -> list[dict[str, Any]]:
            loaded = []
            for name in items:
                path = data_dir / f"{name}.csv"
                if not path.exists():
                    raise FileNotFoundError(f"Missing {path}. Run: python download_ett.py --output-dir {data_dir}")
                # Do not train on a same-named or locally edited CSV: validate the exact
                # official Git blob before parsing and constructing Chronos inputs.
                validate_csv(path, name)
                loaded.append(load_series(path, name, FREQ_MINUTES[frequency]))
            return loaded
        train_series = load_named(names)
        evaluation_series = load_named(eval_names)
        run_dir = resolve_run_dir(out_root, frequency, args.run_name, len(frequencies) > 1)
        if args.preflight_only:
            _, _, train_split = make_fit_inputs(train_series)
            _, _, evaluation_split = make_fit_inputs(evaluation_series)
            result = {
                "preflight_only": True,
                "frequency": frequency,
                "sample_interval_minutes": FREQ_MINUTES[frequency],
                "prediction_length": args.prediction_length,
                "forecast_duration_hours": args.prediction_length * FREQ_MINUTES[frequency] / 60,
                "context_length": args.context_length,
                "training_series": [{"name": x["name"], "rows": x["rows"], "sha256": x["sha256"]}
                                    for x in train_series],
                "evaluation_series": [{"name": x["name"], "rows": x["rows"], "sha256": x["sha256"]}
                                      for x in evaluation_series],
                "training_splits": train_split,
                "evaluation_splits": evaluation_split,
                "output_dir": str(run_dir),
                "model_weights_loaded": False,
                "training_started": False,
            }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            continue
        results.append(train_frequency(
            train_series, frequency, output_dir=run_dir,
            base_model=args.base_model, revision=args.revision, device_map=args.device_map,
            prediction_length=args.prediction_length, context_length=args.context_length,
            num_steps=args.num_steps, batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            finetune_mode=args.finetune_mode, seed=args.seed, test_stride=args.test_stride,
            evaluation_series_list=evaluation_series))
    print(json.dumps({"runs": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
