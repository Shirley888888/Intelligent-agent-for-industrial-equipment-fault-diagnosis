from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from download_ett import validate_csv
from train_ett_corpus import TrainArgs, build_group, load_series, train_one
from diagnostic_agent.ett_corpus_bridge import ETTOilTemperatureBridge
from diagnostic_agent.model_router import TaskAwareModelRouter
from diagnostic_agent.cli import build_agent

ROOT = Path(__file__).resolve().parents[1]


class ETTOilPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_packaged_etth1_is_real_hourly_schema(self):
        path = ROOT / "data" / "ett" / "ETTh1.csv"
        info = validate_csv(path, "ETTh1")
        self.assertEqual(info["rows"], 17420)
        self.assertEqual(info["frequency"], "1h")
        self.assertEqual(info["source_commit"], "1d16c8f4f943005d613b5bc962e9eeb06058cf07")
        self.assertEqual(info["github_blob_sha1"], "a52c4925778c07c1ef1a2cf6fd01594919717d9e")
        manifest = json.loads((ROOT / "data" / "ett" / "source_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_commit"], "1d16c8f4f943005d613b5bc962e9eeb06058cf07")
        self.assertEqual(manifest["expected_files"]["ETTh2.csv"]["github_blob_sha1"],
                         "6e3760778598fdee5208213d3fce122637307eec")
        self.assertEqual(manifest["expected_files"]["ETTm1.csv"]["frequency"], "15min")
        s = load_series(path, "ETTh1", 60)
        self.assertEqual(s["values"].shape, (17420, 7))
        self.assertEqual(s["start"], "2016-07-01T00:00:00")

    def test_train_only_scaler_and_series_split(self):
        path = ROOT / "data" / "ett" / "ETTh1.csv"
        series = load_series(path, "ETTh1", 60)
        cfg = TrainArgs(model="LSTM", input_len=96, pred_len=24,
                        epochs=1, patience=1, batch_size=128,
                        hidden_size=8, num_layers=1, train_stride=32,
                        eval_stride=64)
        group = build_group([series], cfg)
        train_end = group["ranges"]["ETTh1"]["train_end"]
        expected_mean = series["values"][:train_end].mean(axis=0)
        np.testing.assert_allclose(group["mean"], expected_mean, rtol=1e-5, atol=1e-5)
        self.assertEqual(group["train_x"].shape[1:], (96, 7))
        self.assertEqual(group["train_y"].shape[1:], (24,))
        self.assertGreater(len(group["val"]["ETTh1"][0]), 0)
        self.assertGreater(len(group["test"]["ETTh1"][0]), 0)

    def test_lstm_and_tcn_checkpoint_then_agent_episode(self):
        import torch
        rng = np.random.default_rng(19)
        n = 1200
        times = [datetime(2020, 1, 1) + timedelta(hours=i) for i in range(n)]
        t = np.arange(n, dtype=np.float32)
        values = np.stack([
            25 + 2*np.sin(t/24), 12 + np.sin(t/48), 30 + 2*np.cos(t/36),
            8 + np.sin(t/18), 4 + np.cos(t/12), 2 + np.sin(t/72),
            40 + 3*np.sin(t/120) + 0.1*rng.normal(size=n)
        ], axis=1).astype(np.float32)
        series = [{"name": "ETTh-test-fixture", "path": Path("synthetic-test-only.csv"),
                   "values": values, "timestamps": times, "sha256": "test-fixture",
                   "rows": n, "start": times[0].isoformat(), "end": times[-1].isoformat()}]
        # Synthetic data is a tiny unit-test fixture only; it is never packaged as training data.
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "history.csv"
            headers = ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f); writer.writerow(headers)
                for i in range(n):
                    writer.writerow([times[i].isoformat(sep=" "), *values[i].tolist()])
            for model_name in ("LSTM", "TCN"):
                cfg = TrainArgs(model=model_name, input_len=96, pred_len=24,
                                epochs=1, patience=1, batch_size=128,
                                hidden_size=8, num_layers=1, dropout=0.0,
                                tcn_channels=(8, 8), train_stride=16, eval_stride=16)
                model_dir = Path(tmp) / model_name
                train_one(series, "hourly", cfg, model_dir, "cpu")
                bridge = ETTOilTemperatureBridge(model_dir, device="cpu")
                workspace = Path(tmp) / f"agent_{model_name}"
                agent = build_agent(workspace, "offline")
                result = TaskAwareModelRouter(device="cpu").run(agent, {
                    "task_type": "transformer_oil_temperature_forecasting",
                    "model_dir": str(model_dir), "csv": str(csv_path),
                    "equipment_id": "ETT-TEST-TRANSFORMER"})
                self.assertEqual(result["decision"], "human_review")
                self.assertTrue(result["manual_review_requested"])
                self.assertEqual(result["diagnosis"]["prediction"]["target_name"], "OT")
                self.assertEqual(len(result["diagnosis"]["prediction"]["forecast_values"]), 24)
                self.assertTrue(result["episode_persistence"]["stored"])
                self.assertEqual(result["model_input_provenance"]["source"], "ETTOilTemperatureBridge")
                self.assertIn(model_name, result["diagnosis"]["model_version"])

    def test_modified_official_file_is_rejected_by_pinned_github_blob_sha(self):
        source = ROOT / "data" / "ett" / "ETTh1.csv"
        with tempfile.TemporaryDirectory() as tmp:
            changed = Path(tmp) / "ETTh1.csv"
            raw = source.read_bytes()
            token = b"30.5310001373291"
            self.assertIn(token, raw)
            changed.write_bytes(raw.replace(token, b"30.5320001373291", 1))
            with self.assertRaisesRegex(ValueError, "Git blob checksum mismatch"):
                validate_csv(changed, "ETTh1")

    def test_bad_sampling_interval_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"])
                w.writerow(["2020-01-01 00:00:00", 1, 2, 3, 4, 5, 6, 7])
                w.writerow(["2020-01-01 00:15:00", 1, 2, 3, 4, 5, 6, 7])
            with self.assertRaises(ValueError):
                validate_csv(path, "ETTh1")


if __name__ == "__main__":
    unittest.main()
