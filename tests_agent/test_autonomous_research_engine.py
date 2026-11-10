import tempfile
import unittest
from pathlib import Path
import numpy as np

from atrea.research import diagnose_experiment, build_research_hypothesis
from atrea.store import ExperienceStore


class AutonomousResearchEngineTests(unittest.TestCase):
    def test_horizon_diagnostics_identify_late_error_growth(self):
        truth = np.zeros((8, 24, 1), dtype=float)
        pred = np.zeros_like(truth)
        pred[:, :8, :] = 0.1
        pred[:, 8:16, :] = 0.3
        pred[:, 16:, :] = 0.8
        d = diagnose_experiment({"y_true": truth, "y_pred": pred})
        self.assertTrue(d["available"])
        self.assertEqual(len(d["mae_by_horizon"]), 24)
        self.assertEqual(d["worst_horizon"], 17)
        self.assertGreater(d["late_horizon_error_ratio"], 1.0)

    def test_planner_uses_failure_memory_and_paper_methods(self):
        memory = {"experiments": [
            {"model_name": "bad-candidate", "status": "failed", "metrics": {}},
            {"model_name": "baseline", "status": "completed", "metrics": {
                "error_diagnostics": {"bias": 0.2, "late_horizon_error_ratio": 1.0}}}
        ], "patterns": []}
        papers = [{"title": "Reversible Instance Normalization for Time Series",
                   "abstract": "RevIN normalization improves robustness.",
                   "extracted_methods": ["normalization"]}]
        proposal = build_research_hypothesis(papers, memory)
        self.assertTrue(proposal["statement"])
        self.assertIn("bad-candidate", proposal["plan"]["do_not_repeat_failed_models"])
        self.assertIn("normalization", proposal["plan"]["candidate_methods"])
        self.assertTrue(proposal["plan"]["paper_evidence_is_not_performance_evidence"])

    def test_store_returns_failed_experiments_for_learning(self):
        with tempfile.TemporaryDirectory() as td:
            store = ExperienceStore(Path(td) / "memory.sqlite3")
            store.add_experiment({"batch_id": "b", "dataset": "ETTh1", "split_hash": "s",
                "model_name": "failed-model", "config": {}, "hyperparameters": {},
                "metrics": {}, "status": "failed", "memory_scope": "test"})
            got = store.retrieve_experience(memory_scope="test")
            self.assertEqual(len(got["experiments"]), 1)
            self.assertEqual(got["experiments"][0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
