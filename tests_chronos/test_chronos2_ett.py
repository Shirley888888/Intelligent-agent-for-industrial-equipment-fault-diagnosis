from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from train_chronos2_ett import (
    candidate_gate, chronos_input, evaluate_rolling, make_fit_inputs,
    regression_metrics, train_frequency, require_finetune_dependencies,
)
from diagnostic_agent.chronos2_oil_bridge import Chronos2OilTemperatureBridge, sha256_tree
from diagnostic_agent.cli import build_agent


class FakeChronos2Pipeline:
    def predict_quantiles(self, inputs, prediction_length, quantile_levels, **kwargs):
        lower = torch.full((1, prediction_length), 39.5)
        median = torch.full((1, prediction_length), 40.0)
        upper = torch.full((1, prediction_length), 40.5)
        q = torch.stack([lower, median, upper], dim=-1)
        return [q for _ in inputs], [median for _ in inputs]


class Chronos2OilTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_series(self, name="ETTh-test", n=1400):
        t = np.arange(n, dtype=np.float32)
        values = np.stack([
            20 + np.sin(t / 7), 11 + np.cos(t / 9),
            28 + np.sin(t / 13), 8 + np.cos(t / 15),
            4 + np.sin(t / 17), 2 + np.cos(t / 21),
            42 + 1.2*np.sin(t / 24) + 0.2*np.cos(t / 100),
        ], axis=1).astype(np.float32)
        times = [datetime(2020, 1, 1) + timedelta(hours=i) for i in range(n)]
        return {"name": name, "values": values, "timestamps": times,
                "sha256": "test-fixture-sha256", "rows": n,
                "start": times[0].isoformat(), "end": times[-1].isoformat()}

    def test_chronos_inputs_use_ot_target_and_only_past_load_covariates(self):
        s = self.make_series(n=1500)
        item = chronos_input(s["values"], 500)
        self.assertEqual(item["target"].shape, (500,))
        self.assertEqual(set(item["past_covariates"]), {"HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"})
        np.testing.assert_allclose(item["target"], s["values"][:500, -1])
        self.assertNotIn("future_covariates", item)

    def test_train_validation_cutoffs_do_not_reach_test_tail(self):
        a, b = self.make_series("ETTh1", 1500), self.make_series("ETTh2", 1600)
        train, val, ranges = make_fit_inputs([a, b])
        self.assertEqual(len(train), 2)
        self.assertEqual(len(val), 2)
        for idx, series in enumerate((a, b)):
            r = ranges[series["name"]]
            self.assertEqual(len(train[idx]["target"]), r["train_end_exclusive"])
            self.assertEqual(len(val[idx]["target"]), r["fit_validation_end_exclusive"])
            self.assertEqual(r["gate_validation_start_inclusive"], r["fit_validation_end_exclusive"])
            self.assertEqual(r["test_start_inclusive"], r["gate_validation_end_exclusive"])
            self.assertLess(r["gate_validation_end_exclusive"], r["test_end_exclusive"])
        self.assertFalse(np.shares_memory(train[0]["target"], val[0]["target"]))

    def test_gate_rejects_candidate_with_tail_or_station_regression(self):
        common = {"P10_P90_coverage": 0.80, "P10_P90_mean_width_degC": 2.0}
        baseline = {"aggregate": {"MAE_degC": 1.0, "P99_AE_degC": 4.0,
                                  "MaxAE_degC": 6.0, **common},
                    "per_series": {"ETTh1": {"MAE_degC": 1.0, **common},
                                   "ETTh2": {"MAE_degC": 1.0, **common}}}
        bad = {"aggregate": {"MAE_degC": 0.9, "P99_AE_degC": 5.0,
                              "MaxAE_degC": 7.0, **common},
               "per_series": {"ETTh1": {"MAE_degC": 1.0, **common},
                              "ETTh2": {"MAE_degC": 1.1, **common}}}
        result = candidate_gate(baseline, bad)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["aggregate_p99_non_regression"])
        self.assertFalse(result["checks"]["aggregate_maxae_non_regression"])
        self.assertFalse(result["checks"]["each_station_mae_non_regression"])

    def test_gate_rejects_undercovered_or_excessively_wide_intervals(self):
        baseline = {"aggregate": {"MAE_degC": 1.0, "P99_AE_degC": 4.0, "MaxAE_degC": 6.0,
                                  "P10_P90_coverage": 0.80, "P10_P90_mean_width_degC": 2.0},
                    "per_series": {"ETTh1": {"MAE_degC": 1.0, "P10_P90_coverage": 0.80},
                                   "ETTh2": {"MAE_degC": 1.0, "P10_P90_coverage": 0.80}}}
        undercovered = {"aggregate": {"MAE_degC": 0.95, "P99_AE_degC": 3.9, "MaxAE_degC": 6.1,
                                      "P10_P90_coverage": 0.62, "P10_P90_mean_width_degC": 2.1},
                        "per_series": {"ETTh1": {"MAE_degC": 0.95, "P10_P90_coverage": 0.61},
                                       "ETTh2": {"MAE_degC": 0.95, "P10_P90_coverage": 0.63}}}
        result = candidate_gate(baseline, undercovered)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["aggregate_interval_coverage_floor"])
        self.assertFalse(result["checks"]["each_station_interval_coverage_floor"])

        too_wide = {"aggregate": {"MAE_degC": 0.95, "P99_AE_degC": 3.9, "MaxAE_degC": 6.1,
                                   "P10_P90_coverage": 0.82, "P10_P90_mean_width_degC": 3.0},
                    "per_series": {"ETTh1": {"MAE_degC": 0.95, "P10_P90_coverage": 0.82},
                                   "ETTh2": {"MAE_degC": 0.95, "P10_P90_coverage": 0.82}}}
        result = candidate_gate(baseline, too_wide)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["interval_width_not_excessively_wider"])

    def test_metrics_include_tail_errors(self):
        m = regression_metrics(np.array([1, 2, 4, 7]), np.array([1, 2, 3, 4]))
        self.assertAlmostEqual(m["MAE_degC"], 1.0)
        self.assertGreaterEqual(m["P99_AE_degC"], m["P95_AE_degC"])
        self.assertEqual(m["MaxAE_degC"], 3.0)

    def test_chronos2_bridge_writes_probabilistic_forecast_into_agent_episode(self):
        s = self.make_series(n=1400)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_dir = root / "candidate"
            artifact = model_dir / "finetuned-ckpt"
            artifact.mkdir(parents=True)
            (artifact / "mock_weights.bin").write_bytes(b"fake model artifact for bridge test")
            manifest = {
                "task": "transformer_oil_temperature_forecasting",
                "model_artifact": "finetuned-ckpt",
                "model_artifact_sha256_tree": sha256_tree(artifact),
                "frequency": "hourly", "sample_interval_minutes": 60,
                "context_length": 96, "prediction_length": 24,
                "quantile_levels": [0.1, 0.5, 0.9],
                "data_sources": [{"dataset": "ETTh-test", "source_url": "fixture://test"}],
                "model_id": "Chronos2-test-candidate", "base_model_id": "amazon/chronos-2"
            }
            (model_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            csv_path = root / "ETTh-test.csv"
            headers = ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f); writer.writerow(headers)
                for i, ts in enumerate(s["timestamps"]):
                    writer.writerow([ts.isoformat(sep=" "), *s["values"][i].tolist()])
            bridge = Chronos2OilTemperatureBridge(model_dir, pipeline=FakeChronos2Pipeline())
            context = bridge.predict_context(csv_path, equipment_id="TRANSFORMER-TEST")
            self.assertEqual(context["model_prediction"]["prediction_kind"], "probabilistic_regression")
            self.assertEqual(len(context["model_prediction"]["forecast_p10"]), 24)
            self.assertFalse(context["model_prediction"]["confidence_calibrated"])
            agent = build_agent(root / "workspace", "offline")
            report = bridge.run(agent, csv_path, equipment_id="TRANSFORMER-TEST")
            self.assertTrue(report["manual_review_requested"])
            self.assertTrue(report["episode_persistence"]["stored"])
            pred = report["diagnosis"]["prediction"]
            self.assertIn("forecast_p90", pred)


    def test_gate_and_test_rolling_windows_are_disjoint(self):
        series = self.make_series(n=1500)
        gate_metrics, gate_rows = evaluate_rolling(
            FakeChronos2Pipeline(), [series], pred_len=24, test_stride=24,
            context_length=96, segment="gate")
        test_metrics, test_rows = evaluate_rolling(
            FakeChronos2Pipeline(), [series], pred_len=24, test_stride=24,
            context_length=96, segment="test")
        split = make_fit_inputs([series])[2][series["name"]]
        self.assertTrue(gate_rows)
        self.assertTrue(test_rows)
        self.assertTrue(all(split["gate_validation_start_inclusive"] <= row["origin_index"]
                            and row["origin_index"] + 24 <= split["gate_validation_end_exclusive"]
                            for row in gate_rows))
        self.assertTrue(all(split["test_start_inclusive"] <= row["origin_index"]
                            and row["origin_index"] + 24 <= split["test_end_exclusive"]
                            for row in test_rows))
        gate_targets = {(row["origin_index"], row["origin_index"] + 24) for row in gate_rows}
        test_targets = {(row["origin_index"], row["origin_index"] + 24) for row in test_rows}
        self.assertFalse(gate_targets.intersection(test_targets))

    def test_train_frequency_orchestration_records_gate_and_final_test_separately(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        class FakeTrainPipeline(FakeChronos2Pipeline):
            def __init__(self):
                self.model = SimpleNamespace(config=SimpleNamespace(_commit_hash="fake-base-commit"))

            def fit(self, inputs, validation_inputs, output_dir, finetuned_ckpt_name, **kwargs):
                self.fit_kwargs = kwargs
                artifact = Path(output_dir) / finetuned_ckpt_name
                artifact.mkdir(parents=True, exist_ok=True)
                (artifact / "mock_model.bin").write_bytes(b"mock fine-tuned artifact")
                return self

            def predict_quantiles(self, inputs, prediction_length, quantile_levels, **kwargs):
                qs, meds = [], []
                for item in inputs:
                    center = float(np.asarray(item["target"])[-1])
                    median = torch.full((1, prediction_length), center)
                    lower = median - 5.0
                    upper = median + 5.0
                    qs.append(torch.stack([lower, median, upper], dim=-1))
                    meds.append(median)
                return qs, meds

        series = self.make_series(n=1500)
        series["rows"] = len(series["values"])
        series["path"] = Path("fixture.csv")
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "chronos_run"
            fake_pipeline = FakeTrainPipeline()
            with patch.dict("sys.modules", {"peft": object()}), \
                 patch("train_chronos2_ett._load_chronos_pipeline", return_value=fake_pipeline):
                manifest = train_frequency(
                    [series], "hourly", output_dir=out, base_model="amazon/chronos-2",
                    revision="test-revision", device_map="cpu", prediction_length=24,
                    context_length=96, num_steps=2, batch_size=1, learning_rate=1e-5,
                    finetune_mode="lora", seed=7, test_stride=24,
                    gradient_accumulation_steps=3)
            self.assertEqual(manifest["gradient_accumulation_steps"], 3)
            self.assertEqual(manifest["effective_batch_size_per_device"], 3)
            self.assertEqual(fake_pipeline.fit_kwargs["gradient_accumulation_steps"], 3)
            self.assertIn("zero_shot_gate_validation_metrics", manifest)
            self.assertIn("candidate_test_metrics", manifest)
            self.assertEqual(manifest["split_policy"].split(";")[0], "chronological_per_series_60_10_10_20")
            self.assertTrue((out / "gate_validation_predictions.jsonl").exists())
            self.assertTrue((out / "manifest.json").exists())
            self.assertFalse(manifest["agent_activation"])


    def test_lora_refuses_silent_full_finetune_fallback_when_peft_missing(self):
        import sys
        from unittest.mock import patch

        with patch.dict(sys.modules, {"peft": None}):
            with self.assertRaisesRegex(RuntimeError, "refusing to silently fall back"):
                require_finetune_dependencies("lora")
        # Full fine-tuning does not require PEFT, but unknown modes are rejected.
        require_finetune_dependencies("full")
        with self.assertRaises(ValueError):
            require_finetune_dependencies("auto")

    def test_cross_station_holdout_separates_training_and_evaluation_sources(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        class CapturePipeline(FakeChronos2Pipeline):
            def __init__(self):
                self.model = SimpleNamespace(config=SimpleNamespace(_commit_hash="fake-base-commit"))
                self.fit_input_lengths = None

            def fit(self, inputs, validation_inputs, output_dir, finetuned_ckpt_name, **kwargs):
                self.fit_input_lengths = [len(x["target"]) for x in inputs]
                artifact = Path(output_dir) / finetuned_ckpt_name
                artifact.mkdir(parents=True, exist_ok=True)
                (artifact / "mock_model.bin").write_bytes(b"mock cross-station checkpoint")
                return self

            def predict_quantiles(self, inputs, prediction_length, quantile_levels, **kwargs):
                qs, meds = [], []
                for item in inputs:
                    center = float(np.asarray(item["target"])[-1])
                    median = torch.full((1, prediction_length), center)
                    qs.append(torch.stack([median-5.0, median, median+5.0], dim=-1))
                    meds.append(median)
                return qs, meds

        train_station = self.make_series("ETTh1", n=1500)
        eval_station = self.make_series("ETTh2", n=1500)
        for s in (train_station, eval_station):
            s["rows"] = len(s["values"])
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "loso"
            pipe = CapturePipeline()
            with patch.dict("sys.modules", {"peft": object()}), \
                 patch("train_chronos2_ett._load_chronos_pipeline", return_value=pipe):
                manifest = train_frequency(
                    [train_station], "hourly", output_dir=out, base_model="amazon/chronos-2",
                    revision=None, device_map="cpu", prediction_length=24, context_length=96,
                    num_steps=2, batch_size=1, learning_rate=1e-5, finetune_mode="lora",
                    seed=42, test_stride=24, evaluation_series_list=[eval_station])
            self.assertEqual([x["dataset"] for x in manifest["training_data_sources"]], ["ETTh1"])
            self.assertEqual([x["dataset"] for x in manifest["evaluation_data_sources"]], ["ETTh2"])
            self.assertEqual(manifest["evaluation_design"], "cross_station_holdout")
            self.assertEqual(len(pipe.fit_input_lengths), 1)
            self.assertFalse(manifest["agent_activation"])

    def test_checkpoint_integrity_failure_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "candidate"
            artifact = root / "finetuned-ckpt"
            artifact.mkdir(parents=True)
            (artifact / "weights.bin").write_bytes(b"one")
            manifest = {"task": "transformer_oil_temperature_forecasting",
                        "model_artifact": "finetuned-ckpt", "model_artifact_sha256_tree": "wrong",
                        "frequency": "hourly", "sample_interval_minutes": 60,
                        "context_length": 96, "prediction_length": 24}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                Chronos2OilTemperatureBridge(root, pipeline=FakeChronos2Pipeline())


if __name__ == "__main__":
    unittest.main()
