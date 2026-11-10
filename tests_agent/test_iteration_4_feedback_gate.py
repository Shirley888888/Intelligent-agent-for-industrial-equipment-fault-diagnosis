import tempfile
import unittest
from pathlib import Path

from diagnostic_agent.agent import AgentRuntimeConfig, DiagnosticAgent
from diagnostic_agent.providers import OfflinePlanner
from dual_evolution.agent_gate import evaluate_agent_candidate
from dual_evolution.episode_store import EpisodeStore
from dual_evolution.knowledge_catalog import KnowledgeCatalog
from dual_evolution.runtime_guard import RuntimeGuardConfig


class Iteration4FeedbackGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = EpisodeStore(self.root / "episodes.jsonl")
        self.agent = DiagnosticAgent(planner=OfflinePlanner(), knowledge=KnowledgeCatalog(self.root / "kb.jsonl"),
            episode_store=self.store, workspace=self.root,
            config=AgentRuntimeConfig(runtime_guard=RuntimeGuardConfig(calibration_id="TEST-CAL")))

    def tearDown(self):
        self.tmp.cleanup()

    def test_human_feedback_matures_episode(self):
        ctx = {"episode_id": "FB-001", "prediction_time": "2026-11-02T08:00:00+08:00",
            "equipment_id": "P-01", "operating_condition": "normal", "sensor_readings": {"temp": [10]},
            "model": {"model_version": "M0", "activation_id": 1},
            "model_prediction": {"fault_type": "bearing_fault", "confidence": 0.8,
                "confidence_calibrated": True, "ood_score": 0.1, "drift_score": 0.1}}
        self.agent.run(ctx)
        reviewed = self.agent.submit_feedback("FB-001", reviewer="engineer-01",
            source_ref="workorder://WO-2026-01", status="verified",
            ground_truth={"fault_type": "bearing_fault"}, failure_categories=[])
        self.assertEqual(reviewed["status"], "verified")
        self.assertEqual(self.store.get_episode("FB-001")["label"]["status"], "verified")
        self.assertTrue(self.store.verify_integrity()["valid"])

    def test_feedback_requires_auditable_source_ref(self):
        ctx = {"episode_id": "FB-002", "prediction_time": "2026-11-02T08:00:00+08:00",
            "equipment_id": "P-01", "operating_condition": "normal", "sensor_readings": {},
            "model": {"model_version": "M0", "activation_id": 1},
            "model_prediction": {"fault_type": "bearing_fault", "confidence": 0.8,
                "confidence_calibrated": True, "ood_score": 0.1, "drift_score": 0.1}}
        self.agent.run(ctx)
        with self.assertRaises(ValueError):
            self.agent.submit_feedback("FB-002", reviewer="engineer", source_ref="", status="disputed")

    def test_agent_gate_passes_a_safe_manual_escalation_case(self):
        cases = [{"case_id": "OOD-1", "allowed_decisions": ["human_review"],
                  "manual_review_expected": True, "required_tools": ["search_knowledge"],
                  "citations_required": False, "critical": True}]
        outputs = [{"decision": "human_review", "manual_review_requested": True,
                    "tool_calls": [{"name": "search_knowledge", "success": True}], "citations": []}]
        result = evaluate_agent_candidate(cases, outputs, min_case_pass_rate=1.0,
            min_tool_success_rate=1.0, min_citation_coverage=1.0, max_critical_failures=0)
        self.assertEqual(result["status"], "PASS")

    def test_agent_gate_rejects_critical_case_failure(self):
        cases = [{"case_id": "OOD-2", "allowed_decisions": ["human_review"],
                  "manual_review_expected": True, "critical": True}]
        outputs = [{"decision": "provisional_model_result", "manual_review_requested": False,
                    "tool_calls": [], "citations": []}]
        result = evaluate_agent_candidate(cases, outputs, min_case_pass_rate=0.0,
            min_tool_success_rate=0.0, min_citation_coverage=0.0, max_critical_failures=0)
        self.assertEqual(result["status"], "REJECT")
        self.assertIn("critical_case_failure", result["failed_reasons"])

    def test_episode_ledger_detects_tampering(self):
        ctx = {"episode_id": "FB-003", "prediction_time": "2026-11-02T08:00:00+08:00",
            "equipment_id": "P-01", "operating_condition": "normal", "sensor_readings": {},
            "model": {"model_version": "M0", "activation_id": 1},
            "model_prediction": {"fault_type": "bearing_fault", "confidence": 0.8,
                "confidence_calibrated": True, "ood_score": 0.1, "drift_score": 0.1}}
        self.agent.run(ctx)
        path = self.root / "episodes.jsonl"
        text = path.read_text(encoding="utf-8").replace("bearing_fault", "motor_fault")
        path.write_text(text, encoding="utf-8")
        with self.assertRaises(ValueError):
            self.store.verify_integrity()


if __name__ == "__main__":
    unittest.main()
