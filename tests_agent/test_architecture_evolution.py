import unittest

try:
    import torch
except Exception:
    torch = None

from atrea.architecture import (ARCHITECTURE_CATALOG, build_model,
    propose_architectures, run_architecture_sweep, validate_architecture_spec)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ArchitectureEvolutionTests(unittest.TestCase):
    def test_every_registered_architecture_has_expected_io(self):
        x = torch.randn(3, 96, 7)
        for name, base in ARCHITECTURE_CATALOG.items():
            with self.subTest(name=name):
                model = build_model(base)
                y = model(x)
                self.assertEqual(tuple(y.shape), (3, 24))
                self.assertGreater(sum(p.numel() for p in model.parameters()), 0)

    def test_research_methods_propose_executable_architectures(self):
        specs = propose_architectures(["normalization", "patching"], "后段预测误差增长")
        names = {s["model_name"] for s in specs}
        self.assertIn("RevIN_LSTM", names)
        self.assertIn("PatchTransformerLite", names)
        for spec in specs:
            validate_architecture_spec(spec)

    def test_architecture_config_is_allowlisted(self):
        with self.assertRaises(ValueError):
            validate_architecture_spec({"model_name": "LSTM", "config": {"arbitrary_code": "x"}})
        with self.assertRaises(ValueError):
            validate_architecture_spec({"model_name": "LSTM_Attention", "config": {"hidden": 95, "heads": 4}})

    def test_sweep_runs_each_candidate_and_isolates_failures(self):
        calls = []
        def runner(spec, seed):
            calls.append((spec["model_name"], seed))
            if seed == 22:
                raise RuntimeError("simulated candidate failure")
            return {"metrics": {"mae": 1.0 + seed / 1000}}
        records = run_architecture_sweep(
            [{"model_name": "DecompLinear", "config": {}}], runner,
            split_hash="fixed-split", seeds=[11, 22, 33])
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[1]["status"], "failed")
        self.assertEqual(records[0]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
