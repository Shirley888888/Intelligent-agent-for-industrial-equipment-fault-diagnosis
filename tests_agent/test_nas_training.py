import unittest
from pathlib import Path
import torch
from atrea.nas_training import build_etth1_nas_runner
from atrea.nas import build_loss


class NASTrainingAdapterTests(unittest.TestCase):
    def test_real_etth1_one_epoch_smoke(self):
        project_root = Path(__file__).resolve().parents[1]
        if not (project_root / "ETTh1.csv").exists():
            self.skipTest("bundled ETTh1.csv unavailable")
        runner, split_hash = build_etth1_nas_runner(
            epochs=1, batch_size=512, patience=1, device="cpu")
        spec = {"model_name": "NAS_Composed", "config": {
            "branches": [["linear"]], "hidden": 16, "attention": "none",
            "attention_heads": 4, "fusion": "concat", "loss": "mae",
            "optimizer": "adamw", "learning_rate": 1e-3,
            "weight_decay": 1e-4, "scheduler": "none"}}
        result = runner(spec, 11, build_loss("mae"))
        self.assertEqual(len(split_hash), 64)
        self.assertEqual(result["status"], "completed")
        self.assertGreater(result["validation_windows"], 0)
        self.assertTrue(torch.isfinite(torch.tensor(result["metrics"]["mae"])))
        self.assertEqual(result["validation_protocol"],
                         "chronological_60_20_20_train_prefix_scaler_with_disjoint_arch_and_selection_halves")
        self.assertGreater(result["checkpoint_selection_windows"], 0)


    def test_real_etth1_bilevel_darts_updates_architecture_logits(self):
        project_root = Path(__file__).resolve().parents[1]
        if not (project_root / "ETTh1.csv").exists():
            self.skipTest("bundled ETTh1.csv unavailable")
        from atrea.nas import propose_darts_operator
        graph = propose_darts_operator(seed=123, ops=["depthwise_conv", "pointwise_mlp", "spectral_gain"])
        spec = {"model_name": "NAS_Composed", "config": {
            "branches": [[graph, "linear"]], "hidden": 16, "attention": "none",
            "attention_heads": 4, "fusion": "concat", "loss": "mae",
            "optimizer": "adamw", "learning_rate": 1e-3,
            "weight_decay": 1e-4, "scheduler": "none"}}
        runner, _ = build_etth1_nas_runner(epochs=1, batch_size=512, patience=1, device="cpu")
        result = runner(spec, 19, build_loss("mae"))
        self.assertGreater(result["bilevel_architecture_updates"], 0)
        self.assertEqual(result["bilevel_mode"], "first_order_train_weights_arch_validation_logits")
        self.assertGreater(result["architecture_validation_windows"], 0)
        self.assertGreater(result["checkpoint_selection_windows"], 0)


if __name__ == "__main__":
    unittest.main()
