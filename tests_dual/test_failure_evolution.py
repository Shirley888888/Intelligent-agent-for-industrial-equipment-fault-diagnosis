import tempfile
import unittest
from pathlib import Path

from dual_evolution.failure_evolution import (
    FailureDrivenEvolution, FailureEvolutionConfig, FailureExperienceLedger,
    diagnose_candidate_failure,
)


class TestFailureDrivenEvolution(unittest.TestCase):
    def test_rejected_candidate_is_recorded_and_next_candidate_can_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = FailureExperienceLedger(Path(temp) / "failure_ledger.jsonl")
            engine = FailureDrivenEvolution(ledger, FailureEvolutionConfig(
                max_attempts_per_candidate_family=3, max_total_attempts=3,
                max_total_training_seconds=30))
            calls = []

            def train(candidate, attempt):
                calls.append(candidate["architecture"])
                return {"status": "completed", "train_seconds": 1.0,
                        "metrics": {"mae": 1.5 if attempt == 1 else 1.2}}

            def gate(candidate, training):
                if candidate["architecture"] == "transformer":
                    return {"status": "REJECT", "failures": ["global.mae"],
                            "insufficient_evidence": []}
                return {"status": "PASS", "failures": [], "insufficient_evidence": []}

            def propose(candidate, diagnosis, history):
                self.assertIn("prediction_quality_regression", diagnosis["categories"])
                self.assertEqual(history[-1]["outcome"], "REJECT")
                return {"architecture": "cnn_transformer"}

            result = engine.run({"architecture": "transformer"},
                                train_and_evaluate=train, gate_evaluate=gate,
                                propose_next=propose, candidate_family="oil_temp")
            self.assertEqual(result["status"], "ELIGIBLE_FOR_SHADOW")
            self.assertEqual(calls, ["transformer", "cnn_transformer"])
            self.assertFalse(result["production_model_changed"])
            self.assertEqual(result["attempts"][0]["outcome"], "REJECT")
            self.assertEqual(result["attempts"][1]["outcome"], "PASS")
            self.assertTrue(ledger.read())

    def test_review_required_never_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            engine = FailureDrivenEvolution(FailureExperienceLedger(Path(temp) / "l.jsonl"))
            result = engine.run({"arch": "a"},
                train_and_evaluate=lambda c, a: {"status": "completed", "train_seconds": 0.1},
                gate_evaluate=lambda c, t: {"status": "REVIEW_REQUIRED",
                                            "failures": [], "insufficient_evidence": ["global.p99"]})
            self.assertEqual(result["status"], "STOPPED_RETRY_LIMIT")
            self.assertIsNone(result["eligible_candidate"])

    def test_duplicate_next_configuration_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            engine = FailureDrivenEvolution(FailureExperienceLedger(Path(temp) / "l.jsonl"),
                FailureEvolutionConfig(max_attempts_per_candidate_family=3))
            result = engine.run({"arch": "a"},
                train_and_evaluate=lambda c, a: {"status": "completed", "train_seconds": 0.1},
                gate_evaluate=lambda c, t: {"status": "REJECT", "failures": ["global.mae"]},
                propose_next=lambda c, d, h: {"arch": "a"})
            self.assertEqual(result["status"], "STOPPED_DUPLICATE_CONFIGURATION")
            self.assertEqual(result["attempt_count"], 1)

    def test_training_exception_is_recorded_and_does_not_promote(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = FailureExperienceLedger(Path(temp) / "l.jsonl")
            engine = FailureDrivenEvolution(ledger)
            result = engine.run({"arch": "broken"},
                train_and_evaluate=lambda c, a: (_ for _ in ()).throw(RuntimeError("oom")),
                gate_evaluate=lambda c, t: {"status": "PASS"})
            self.assertEqual(result["status"], "STOPPED_RETRY_LIMIT")
            self.assertFalse(result["production_model_changed"])
            self.assertIn("training_or_runtime_error", ledger.read()[0]["payload"]["diagnosis"]["categories"])

    def test_ledger_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "l.jsonl"
            ledger = FailureExperienceLedger(path)
            ledger.append({"x": 1})
            path.write_text(path.read_text().replace('"x":1', '"x":2'), encoding="utf-8")
            with self.assertRaises(ValueError):
                ledger.read()

    def test_diagnosis_does_not_claim_causality(self):
        diagnosis = diagnose_candidate_failure({"metrics": {"mae": 2.0}},
            {"status": "REJECT", "failures": ["global.mae"], "insufficient_evidence": []})
        self.assertEqual(diagnosis["root_cause_confidence"], "evidence_only_not_causal_proof")


if __name__ == "__main__":
    unittest.main(verbosity=2)
