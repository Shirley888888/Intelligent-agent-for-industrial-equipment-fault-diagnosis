from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from diagnostic_agent.agent import AgentRuntimeConfig, DiagnosticAgent
from diagnostic_agent.industrial_bridge import VersionedIndustrialFaultBridge
from diagnostic_agent.industrial_fault import IndustrialFaultTrainConfig, train_industrial_fault_model
from diagnostic_agent.model_router import TaskAwareModelRouter
from diagnostic_agent.providers import OfflinePlanner
from dual_evolution.episode_store import EpisodeStore
from dual_evolution.knowledge_catalog import KnowledgeCatalog
from dual_evolution.runtime_guard import RuntimeGuardConfig

ROOT = Path(__file__).resolve().parents[1]


def make_agent(workspace: Path, calibration_id: str = "UNSET") -> DiagnosticAgent:
    return DiagnosticAgent(
        planner=OfflinePlanner(),
        knowledge=KnowledgeCatalog(workspace / "knowledge.jsonl"),
        episode_store=EpisodeStore(workspace / "episodes.jsonl"),
        workspace=workspace,
        config=AgentRuntimeConfig(runtime_guard=RuntimeGuardConfig(calibration_id=calibration_id)),
    )


def make_dataset(path: Path):
    # Deterministic synthetic fixture used only to test shapes and wiring.
    rng = np.random.default_rng(20261102)
    n_train, n_val, seq_len, n_features = 90, 45, 12, 4
    y_train = np.tile(np.arange(3, dtype=np.int64), n_train // 3)
    y_val = np.tile(np.arange(3, dtype=np.int64), n_val // 3)
    x_train = rng.normal(0, 0.15, (n_train, seq_len, n_features)).astype(np.float32)
    x_val = rng.normal(0, 0.15, (n_val, seq_len, n_features)).astype(np.float32)
    y_cal_fit = np.tile(np.arange(3, dtype=np.int64), 15)
    y_cal_eval = np.tile(np.arange(3, dtype=np.int64), 15)
    x_cal_fit = rng.normal(0, 0.15, (len(y_cal_fit), seq_len, n_features)).astype(np.float32)
    x_cal_eval = rng.normal(0, 0.15, (len(y_cal_eval), seq_len, n_features)).astype(np.float32)
    for i, label in enumerate(y_train):
        x_train[i, :, int(label)] += 1.8
    for i, label in enumerate(y_val):
        x_val[i, :, int(label)] += 1.8
    for i, label in enumerate(y_cal_fit):
        x_cal_fit[i, :, int(label)] += 1.8
    for i, label in enumerate(y_cal_eval):
        x_cal_eval[i, :, int(label)] += 1.8
    def targets(labels):
        return {
            "anomaly": (labels != 0).astype(np.float32),
            "rul": np.where(labels == 0, 80.0, np.where(labels == 1, 24.0, 12.0)).astype(np.float32),
            "amp": np.where(labels == 0, 0.0, np.where(labels == 1, 0.4, 0.8)).astype(np.float32),
        }
    tr, va = targets(y_train), targets(y_val)
    np.savez(path, X_train=x_train, X_val=x_val,
             y_fault_train=y_train, y_fault_val=y_val,
             y_anomaly_train=tr["anomaly"], y_anomaly_val=va["anomaly"],
             y_rul_hours_train=tr["rul"], y_rul_hours_val=va["rul"],
             y_fault_amplitude_train=tr["amp"], y_fault_amplitude_val=va["amp"],
             X_cal_fit=x_cal_fit, y_fault_cal_fit=y_cal_fit,
             X_cal_eval=x_cal_eval, y_fault_cal_eval=y_cal_eval)
    return x_val, y_val


def train_fixture(root: Path, model_name: str = "LSTM"):
    dataset = root / "fault_dataset.npz"
    x_val, y_val = make_dataset(dataset)
    cfg = IndustrialFaultTrainConfig(
        model_name=model_name,
        fault_classes=["normal", "bearing_fault", "imbalance_fault"],
        n_features=4, input_len=12,
        hidden_size=16, num_layers=1, tcn_channels=(8, 16), kernel_size=3,
        dropout=0.0, epochs=3, patience=2, batch_size=15, learning_rate=0.01,
        device="cpu", calibration_min_samples=20,
        # The synthetic fixture tests wiring, not model quality. The production defaults are stricter.
        min_val_accuracy=0.0, min_class_recall=0.0, min_class_support=1,
        max_rul_mae=10000.0, max_amplitude_mae=10000.0, min_anomaly_auroc=0.0,
    )
    model_root = root / f"registry_{model_name.lower()}"
    result = train_industrial_fault_model(dataset, cfg, model_root)
    return x_val, y_val, result, model_root


class DualTaskModelTests(unittest.TestCase):
    def test_industrial_lstm_trains_loads_and_returns_multitask_outputs(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch unavailable")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            x_val, _, result, model_root = train_fixture(root, "LSTM")
            self.assertTrue(result["active"])
            bridge = VersionedIndustrialFaultBridge(model_root, device="cpu")
            context = bridge.predict_context(
                x_val[0], equipment_id="PUMP-TEST-01", prediction_time="2026-11-02T10:00:00+08:00",
                operating_condition="steady_load", feature_names=["vib_x", "vib_y", "temp", "current"])
            prediction = context["model_prediction"]
            self.assertIn(prediction["fault_type"], {"normal", "bearing_fault", "imbalance_fault"})
            self.assertEqual(len(prediction["fault_probabilities"]), 3)
            self.assertIsNotNone(prediction["anomaly_score"])
            self.assertIsNotNone(prediction["rul_hours"])
            self.assertIsNotNone(prediction["fault_amplitude"])
            self.assertEqual(context["model"]["model_name"], "LSTM")

    def test_industrial_tcn_trains_loads_and_returns_fault_class(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch unavailable")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            x_val, _, result, model_root = train_fixture(root, "TCN")
            self.assertTrue(result["active"])
            bridge = VersionedIndustrialFaultBridge(model_root, device="cpu")
            context = bridge.predict_context(
                x_val[1], equipment_id="PUMP-TEST-02", prediction_time="2026-11-02T10:00:00+08:00",
                operating_condition="variable_load")
            self.assertIn(context["model_prediction"]["fault_type"],
                          {"normal", "bearing_fault", "imbalance_fault"})
            self.assertEqual(context["task_type"], "industrial_fault_diagnosis")

    def test_missing_ood_or_drift_signals_force_human_review(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            x_val, _, _, model_root = train_fixture(root, "LSTM")
            bridge = VersionedIndustrialFaultBridge(model_root, device="cpu")
            context = bridge.predict_context(
                x_val[0], equipment_id="PUMP-TEST-03", prediction_time="2026-11-02T10:00:00+08:00",
                operating_condition="steady_load")
            agent = make_agent(root / "agent", calibration_id=context["model_prediction"].get("calibration_id") or "UNSET")
            report = agent.run(context)
            self.assertEqual(report["decision"], "human_review")
            self.assertIn("missing_runtime_signal", report["runtime_safety"]["reasons"])

    def test_calibration_id_mismatch_forces_human_review(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            x_val, _, _, model_root = train_fixture(root, "LSTM")
            bridge = VersionedIndustrialFaultBridge(
                model_root, device="cpu",
                runtime_signal_provider=lambda history, meta: {"ood_score": 0.1, "drift_score": 0.1})
            context = bridge.predict_context(
                x_val[0], equipment_id="PUMP-TEST-04", prediction_time="2026-11-02T10:00:00+08:00",
                operating_condition="steady_load")
            agent = make_agent(root / "agent", calibration_id="WRONG-CALIBRATION-ID")
            report = agent.run(context)
            self.assertEqual(report["decision"], "human_review")
            self.assertIn("confidence_not_calibrated", report["runtime_safety"]["reasons"])

    def test_missing_independent_calibration_set_prevents_activation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = root / "fault_dataset.npz"
            x_val, _ = make_dataset(dataset)
            # Rebuild the NPZ without calibration arrays. A high classification score
            # alone is not enough to activate a production industrial checkpoint.
            with np.load(dataset, allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files
                          if not k.startswith("X_cal_") and not k.startswith("y_fault_cal_")}
            np.savez(dataset, **arrays)
            cfg = IndustrialFaultTrainConfig(
                model_name="LSTM", fault_classes=["normal", "bearing_fault", "imbalance_fault"],
                n_features=4, input_len=12, hidden_size=16, num_layers=1,
                epochs=3, patience=2, batch_size=15, learning_rate=0.01,
                device="cpu", min_val_accuracy=0.0, min_class_recall=0.0,
                min_class_support=1, require_confidence_calibration=True,
                max_rul_mae=10000.0, max_amplitude_mae=10000.0, min_anomaly_auroc=0.0,
            )
            output = root / "uncalibrated_registry"
            result = train_industrial_fault_model(dataset, cfg, output)
            self.assertFalse(result["active"])
            self.assertFalse(result["metadata"]["confidence_calibrated"])
            self.assertFalse((output / "current.json").exists())
            self.assertFalse(result["release_gate"]["confidence_calibration_ok"])

    def test_auxiliary_targets_require_explicit_release_thresholds(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch unavailable")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = root / "fault_dataset.npz"
            make_dataset(dataset)
            cfg = IndustrialFaultTrainConfig(
                model_name="LSTM", fault_classes=["normal", "bearing_fault", "imbalance_fault"],
                n_features=4, input_len=12, hidden_size=8, num_layers=1,
                epochs=2, patience=1, batch_size=30, device="cpu",
                calibration_min_samples=20, min_val_accuracy=0.0,
                min_class_recall=0.0, min_class_support=1,
                # Auxiliary targets are present in the dataset, but their release limits are deliberately absent.
                max_rul_mae=None, max_amplitude_mae=None, min_anomaly_auroc=None,
            )
            output = root / "missing_auxiliary_threshold_registry"
            result = train_industrial_fault_model(dataset, cfg, output)
            self.assertFalse(result["active"])
            self.assertFalse(result["release_gate"]["rul_gate_ok"])
            self.assertFalse(result["release_gate"]["amplitude_gate_ok"])
            self.assertFalse(result["release_gate"]["anomaly_gate_ok"])
            self.assertFalse((output / "current.json").exists())
            self.assertTrue((output / "candidates" / "C0" / "manifest.json").exists())

    def test_default_release_gate_blocks_low_quality_model(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch unavailable")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = root / "fault_dataset.npz"
            make_dataset(dataset)
            cfg = IndustrialFaultTrainConfig(
                model_name="LSTM", fault_classes=["normal", "bearing_fault", "imbalance_fault"],
                n_features=4, input_len=12, hidden_size=8, num_layers=1,
                epochs=1, patience=1, batch_size=30, device="cpu",
                min_val_accuracy=0.99, min_class_recall=0.99, min_class_support=5,
                max_rul_mae=10000.0, max_amplitude_mae=10000.0, min_anomaly_auroc=0.0,
            )
            output = root / "candidate_registry"
            result = train_industrial_fault_model(dataset, cfg, output)
            self.assertFalse(result["active"])
            self.assertFalse((output / "current.json").exists())
            self.assertTrue((output / "candidates" / "C0" / "manifest.json").exists())
            with self.assertRaises(FileNotFoundError):
                VersionedIndustrialFaultBridge(output, device="cpu")

    def test_unified_router_sends_forecast_and_fault_tasks_to_same_episode_store(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch unavailable")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, _, _, fault_root = train_fixture(root, "LSTM")
            forecast_root = ROOT / "run_gpu_smoke"
            csv_path = ROOT / "ETTh1.csv"
            if not (forecast_root / "current.json").exists() or not csv_path.exists():
                self.skipTest("Bundled ETTh1 LSTM smoke checkpoint unavailable")
            # Reuse one Agent and its one EpisodeStore for both task types.
            workspace = root / "shared_agent_workspace"
            agent = make_agent(workspace)
            router = TaskAwareModelRouter(device="cpu")
            forecast_config = {
                "task_type": "etth1_forecast", "model_root": str(forecast_root),
                "csv": str(csv_path), "equipment_id": "ETTH1-FORECAST-01",
                "episode_id": "DUAL-TASK-FORECAST-001",
            }
            # The industrial context is directly sourced from the independently trained checkpoint.
            sample = make_dataset(root / "router_fault_input.npz")[0][0]
            input_json = root / "fault_input.json"
            input_json.write_text(json.dumps({
                "equipment_id": "PUMP-DUAL-01", "prediction_time": "2026-11-02T10:00:00+08:00",
                "operating_condition": "steady_load", "feature_names": ["f1", "f2", "f3", "f4"],
                "history": sample.tolist(), "episode_id": "DUAL-TASK-FAULT-001",
            }), encoding="utf-8")
            fault_config = {"task_type": "industrial_fault_diagnosis", "model_root": str(fault_root),
                            "input_json": str(input_json)}
            bundle_report = router.run(agent, {"task_type": "multi_task_bundle",
                                               "tasks": [forecast_config, fault_config]})
            episodes = agent.episode_store.list_episodes()
            self.assertEqual(bundle_report["task_count"], 2)
            self.assertEqual(len(episodes), 2)
            self.assertEqual({x["task_type"] for x in episodes},
                             {"time_series_forecasting", "industrial_fault_diagnosis"})
            self.assertTrue(all(x["episode_persistence"]["stored"] for x in bundle_report["results"]))

    def test_checksum_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, _, _, model_root = train_fixture(root, "LSTM")
            ckpt = model_root / "versions" / "M0" / "industrial_fault.pt"
            ckpt.write_bytes(ckpt.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                VersionedIndustrialFaultBridge(model_root, device="cpu")


if __name__ == "__main__":
    unittest.main()
