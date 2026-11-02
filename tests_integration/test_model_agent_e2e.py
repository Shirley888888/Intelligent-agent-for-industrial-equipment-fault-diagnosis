import csv
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from diagnostic_agent.agent import AgentRuntimeConfig, DiagnosticAgent
from diagnostic_agent.model_bridge import VersionedForecastModelBridge, StructuredPredictorBridge
from diagnostic_agent.providers import OfflinePlanner
from dual_evolution.episode_store import EpisodeStore
from dual_evolution.knowledge_catalog import KnowledgeCatalog
from dual_evolution.runtime_guard import RuntimeGuardConfig
from minimal_loop import (FEATURES, TrainConfig, VersionManager, digest, iso,
                          make_version, train_candidate_tool)


ROOT = Path(__file__).resolve().parents[1]


def build_agent(workspace: Path) -> DiagnosticAgent:
    return DiagnosticAgent(
        planner=OfflinePlanner(),
        knowledge=KnowledgeCatalog(workspace / "knowledge.jsonl"),
        episode_store=EpisodeStore(workspace / "episodes.jsonl"),
        workspace=workspace,
        config=AgentRuntimeConfig(runtime_guard=RuntimeGuardConfig(calibration_id="UNSET")),
    )


class ModelAgentEndToEndTests(unittest.TestCase):
    def test_real_versioned_lstm_checkpoint_runs_through_agent(self):
        model_root = ROOT / "run_gpu_smoke"
        csv_path = ROOT / "ETTh1.csv"
        if not (model_root / "current.json").exists() or not csv_path.exists():
            self.skipTest("Bundled LSTM checkpoint or ETTh1 replay data is absent")
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            agent = build_agent(workspace)
            bridge = VersionedForecastModelBridge(model_root, device="cpu")
            report = bridge.run(agent, csv_path, equipment_id="ETTH1-LSTM-REPLAY",
                                operating_condition="historical_dataset_replay",
                                episode_id="E2E-LSTM-001")
            self.assertEqual(report["diagnosis"]["fault_type"], None)
            self.assertEqual(report["decision"], "human_review")
            forecast = report["diagnosis"]["prediction"]["forecast_values"]
            self.assertEqual(len(forecast), 24)
            self.assertTrue(np.isfinite(np.asarray(forecast)).all())
            self.assertEqual(report["model_input_provenance"]["source"], "VersionedForecastModelBridge")
            episode = agent.episode_store.get_episode("E2E-LSTM-001")
            self.assertEqual(episode["model"]["model_name"], "LSTM")
            self.assertEqual(episode["model"]["model_version"], "M0")
            self.assertEqual(episode["source_data"]["history_rows"], 96)
            self.assertIsNone(episode["model"]["prediction"]["fault_type"])

    def test_bundled_etth1_trained_tcn_checkpoint_runs_through_agent(self):
        model_root = ROOT / "run_tcn_agent_etth1"
        csv_path = ROOT / "ETTh1.csv"
        if not (model_root / "current.json").exists() or not csv_path.exists():
            self.skipTest("Bundled ETTh1-trained TCN checkpoint or replay data is absent")
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            agent = build_agent(workspace)
            bridge = VersionedForecastModelBridge(model_root, device="cpu")
            report = bridge.run(agent, csv_path, equipment_id="ETTH1-TCN-REPLAY",
                                operating_condition="historical_dataset_replay",
                                episode_id="E2E-TCN-BUNDLED-001")
            self.assertEqual(report["decision"], "human_review")
            self.assertIsNone(report["diagnosis"]["fault_type"])
            self.assertEqual(len(report["diagnosis"]["prediction"]["forecast_values"]), 24)
            episode = agent.episode_store.get_episode("E2E-TCN-BUNDLED-001")
            self.assertEqual(episode["model"]["model_name"], "TCN")
            self.assertEqual(episode["model"]["model_version"], "M0")
            self.assertEqual(episode["model"]["activation_id"], 0)

    def test_structured_fault_model_output_is_passed_through_not_inferred(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            agent = build_agent(workspace)
            bridge = StructuredPredictorBridge(
                lambda history: {"fault_type": "bearing_fault_v2", "confidence": 0.91,
                                 "confidence_calibrated": True, "ood_score": 0.08,
                                 "drift_score": 0.04, "rul_hours": 31.5,
                                 "fault_amplitude": 0.37},
                model_name="TCN", model_version="FAULT-TCN-M7", activation_id=7,
                checkpoint_sha256="a" * 64)
            context = bridge.predict_context(
                np.ones((96, 7)), equipment_id="PUMP-REAL-ADAPTER",
                prediction_time="2026-11-02T08:00:00+08:00",
                operating_condition="steady_load",
                sensor_readings={"bearing_temp_c": [65.0, 66.0]},
                episode_id="E2E-STRUCTURED-FAULT-001")
            report = agent.run(context)
            self.assertEqual(report["diagnosis"]["fault_type"], "bearing_fault_v2")
            self.assertEqual(report["diagnosis"]["model_version"], "FAULT-TCN-M7")
            self.assertEqual(report["diagnosis"]["prediction"]["rul_hours"], 31.5)

    def test_trained_tcn_checkpoint_runs_through_agent(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is unavailable")
        rng = np.random.default_rng(2026)
        n = 720
        t = np.arange(n, dtype=float)[:, None]
        values = np.concatenate([
            0.3 * np.sin(t / (17.0 + j) + j * 0.2) + rng.normal(0, 0.03, (n, 1))
            for j in range(7)
        ], axis=1)
        times = [datetime(2020, 1, 1) + timedelta(hours=i) for i in range(n)]
        as_of = times[-1] + timedelta(hours=1)
        data_version = make_version(values, times, as_of, [(0, 300)],
                                    (420, 540), (300, 420), (540, 720))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            candidate = train_candidate_tool(
                data_version,
                TrainConfig(backend="torch", model_name="TCN", tcn_channels=(4, 8),
                            kernel_size=3, dropout=0.0, epochs=1, patience=1,
                            batch_size=8, train_stride=24, device="cpu"),
                values, times, root / "candidate")
            model_root = root / "model_root"
            version_dir = model_root / "versions" / "M0"
            version_dir.mkdir(parents=True)
            checkpoint = version_dir / "M0_checkpoint.pt"
            shutil.copy2(candidate["checkpoint_path"], checkpoint)
            manifest = {
                "model_version": "M0", "checkpoint_filename": checkpoint.name,
                "checkpoint_sha256": digest(checkpoint),
                "train_config": candidate["train_config"],
                "manifest": candidate["manifest"],
                "baseline_mae": candidate["validation_metrics"]["MAE"],
                "activated_as_of": data_version.as_of,
            }
            (version_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            pointer = {"model_version": "M0", "activation_id": 1,
                       "activated_as_of": data_version.as_of, "action": "bootstrap",
                       "baseline_mae": candidate["validation_metrics"]["MAE"]}
            model_root.mkdir(parents=True, exist_ok=True)
            (model_root / "current.json").write_text(json.dumps(pointer), encoding="utf-8")

            csv_path = root / "tcn_input.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["date"] + FEATURES)
                for time_value, row in zip(times, values):
                    writer.writerow([iso(time_value)] + row.tolist())

            # Also verify the loaded object is the actual TCN class before Agent execution.
            loaded, _ = VersionManager(model_root, device="cpu").load("M0")
            self.assertEqual(type(loaded.net).__name__, "TCNModel")
            agent = build_agent(root / "agent_workspace")
            bridge = VersionedForecastModelBridge(model_root, device="cpu")
            report = bridge.run(agent, csv_path, equipment_id="SYNTHETIC-TCN-REPLAY",
                                operating_condition="controlled_test", episode_id="E2E-TCN-001")
            self.assertEqual(report["decision"], "human_review")
            self.assertIsNone(report["diagnosis"]["fault_type"])
            self.assertEqual(len(report["diagnosis"]["prediction"]["forecast_values"]), 24)
            episode = agent.episode_store.get_episode("E2E-TCN-001")
            self.assertEqual(episode["model"]["model_name"], "TCN")
            self.assertEqual(episode["model"]["activation_id"], 1)


if __name__ == "__main__":
    unittest.main()
