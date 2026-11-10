"""modify:torch-version-checks - version-switching invariants on the torch backend.

These engineering checks run against the actual PyTorch installation (GPU when
available, CPU otherwise). Gate scores used here are controlled test inputs,
not model evaluation results. All output directories live in a TemporaryDirectory.
"""
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # the NumPy reference backend stays usable without torch
    torch = None

from minimal_loop import (ForecastModel, GateConfig, TrainConfig, VersionManager,
                          digest, load_csv, model_gate, model_gate_tool, simulate,
                          validate_candidate)

PREFIX = 3048


def torch_device() -> str:
    return "cuda:0" if torch is not None and torch.cuda.is_available() else "cpu"


def tiny_config(device: str) -> TrainConfig:
    return TrainConfig(backend="torch", model_name="LSTM", device=device, epochs=1,
                       patience=1, train_stride=24, hidden_size=32, num_layers=1)


def tiny_simulate(values: np.ndarray, times: list, out: str | Path, device: str) -> dict:
    return simulate(values[:PREFIX], times[:PREFIX], out, tiny_config(device),
                    train_end=1200, online_start=3000, monitor_n=1, patience=2,
                    max_episodes=2, exercise_at=2)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchVersionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch_device()
        cls.data, cls.times = load_csv(Path(__file__).with_name("ETTh1.csv"))
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.original = cls.root/"original"
        cls.summary = tiny_simulate(cls.data, cls.times, cls.original, cls.device)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def candidate(self, name: str = "C0") -> dict:
        return json.loads((self.original/"candidates"/name/"manifest.json").read_text(encoding="utf-8"))

    def controlled_gate(self, registry: VersionManager, candidate: dict) -> dict:
        current = registry.current()
        model, _ = registry.load(current["model_version"])
        return {**model_gate(1.0, 0.90, 1.0, 1.0),
                "incumbent_version": current["model_version"],
                "incumbent_activation_id": current["activation_id"],
                "incumbent_checkpoint_sha256": digest(model.checkpoint),
                "candidate_id": candidate["candidate_id"],
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "snapshot_sha256": candidate["manifest"]["snapshot_sha256"],
                "gate_config": vars(GateConfig()),
                "evaluated_as_of": candidate["trained_as_of"]}

    def test_bootstrap_summary_and_version_binding(self):
        self.assertEqual((self.summary["backend"], self.summary["model_name"]), ("torch", "LSTM"))
        self.assertEqual(self.summary["candidate_errors"], [])
        # On this tiny prefix the exercise candidate may legitimately pass the
        # gate with real scores, so the final version is M0 or M1.
        current = self.summary["current"]
        self.assertIn(current["model_version"], ("M0", "M1"))
        self.assertEqual(current["action"], "bootstrap" if current["model_version"] == "M0" else "upgrade")
        manager = VersionManager(self.original, self.device)
        self.assertEqual(manager.current(), current)
        model, manifest = manager.load(current["model_version"])
        self.assertTrue(manifest["checkpoint_filename"].endswith(".pt"))
        self.assertEqual(manifest["checkpoint_sha256"], digest(model.checkpoint))
        prediction = model.predict_batch(np.zeros((2, 96, 7)))
        self.assertEqual(prediction.shape, (2, 24))
        self.assertTrue(np.isfinite(prediction).all())

    def test_same_seed_reproduces_training(self):
        repeat = tiny_simulate(self.data, self.times, self.root/"repeat", self.device)
        self.assertEqual(repeat["prequential_mae"], self.summary["prequential_mae"])
        self.assertEqual(self.candidate()["validation_metrics"],
                         json.loads((self.root/"repeat"/"candidates"/"C0"/"manifest.json")
                                    .read_text(encoding="utf-8"))["validation_metrics"])
        first = torch.load(self.original/"candidates"/"C0"/"checkpoint.pt",
                           map_location="cpu", weights_only=True)
        second = torch.load(self.root/"repeat"/"candidates"/"C0"/"checkpoint.pt",
                            map_location="cpu", weights_only=True)
        self.assertEqual(first["metadata"], second["metadata"])
        self.assertEqual(first["mean"], second["mean"])
        self.assertEqual(first["scale"], second["scale"])
        self.assertEqual(set(first["model_state"]), set(second["model_state"]))
        for key, tensor in first["model_state"].items():
            self.assertTrue(torch.equal(tensor, second["model_state"][key]), key)

    def test_gate_tool_produces_valid_json_on_torch_checkpoints(self):
        candidate = self.candidate("C1")
        end = candidate["manifest"]["observed_end"]
        incumbent, _ = VersionManager(self.original, self.device).load("M0")
        gate = model_gate_tool(candidate, incumbent, "M0", self.data[:end],
                               self.times[:end], self.device)
        json.dumps(gate, allow_nan=False)
        self.assertIsInstance(gate["recent"]["candidate"]["MAE"], float)
        self.assertIsInstance(gate["allow_upgrade"], bool)

    def test_upgrade_rollback_stale_gate_and_corruption(self):
        registry = VersionManager(self.root/"registry", self.device)
        candidate = self.candidate()
        now = "2099-01-01T00:00:00"
        registry.publish(candidate, None, now, 1.0)
        gate = self.controlled_gate(registry, candidate)
        registry.publish(candidate, gate, now, 0.90)
        self.assertEqual(registry.current()["model_version"], "M1")
        registry.activate("M0", now)
        before = registry.current()
        with self.assertRaises(ValueError):
            registry.publish(candidate, gate, now, 0.90)  # gate bound to a prior activation
        self.assertEqual(registry.current(), before)
        self.assertFalse((registry.root/"versions"/"M2").exists())
        registry.publish(candidate, self.controlled_gate(registry, candidate), now, 0.90)
        self.assertEqual(registry.current()["model_version"], "M2")
        registry.activate("M1", now)
        before = registry.current()
        (registry.root/"versions"/"M0"/"M0_checkpoint.pt").write_bytes(b"broken")
        with self.assertRaises(ValueError):
            registry.activate("M0", now)
        self.assertEqual(registry.current(), before)

    def test_validate_candidate_rejects_tampered_sidecar(self):
        candidate = self.candidate()
        validate_candidate(candidate, self.device)
        mutations = [lambda c: c["train_config"].update(epochs=99),
                     lambda c: c["manifest"].update(observed_end=c["manifest"]["observed_end"]+1),
                     lambda c: c.update(checkpoint_sha256="0"*64)]
        for mutate in mutations:
            bad = deepcopy(candidate)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_candidate(bad, self.device)

    def test_predictions_are_stable_across_devices(self):
        if self.device == "cpu":
            self.skipTest("CUDA device unavailable; cross-device check skipped")
        checkpoint = self.original/"versions"/"M0"/"M0_checkpoint.pt"
        cpu_model = ForecastModel(checkpoint, "cpu")
        gpu_model = ForecastModel(checkpoint, self.device)
        history = self.data[2904:3000]
        once = gpu_model.predict_batch(history[None])
        np.testing.assert_array_equal(once, gpu_model.predict_batch(history[None]))
        np.testing.assert_allclose(cpu_model.predict_batch(history[None]), once,
                                   rtol=1e-4, atol=1e-4)

    def test_unavailable_cuda_device_is_rejected(self):
        if torch.cuda.is_available():
            self.skipTest("CUDA is available; rejection path not reachable")
        with self.assertRaises(RuntimeError):
            ForecastModel(self.original/"candidates"/"C0"/"checkpoint.pt", "cuda:0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
