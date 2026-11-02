from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dual_evolution.agent_gate import compare_agent_candidates, evaluate_agent_candidate
from dual_evolution.demo_v0 import run_demo
from dual_evolution.episode_store import EpisodeStore
from dual_evolution.knowledge_catalog import KnowledgeCatalog
from dual_evolution.model_gate import ModelGateConfig, evaluate_model_candidate
from dual_evolution.release_registry import JointReleaseRegistry
from dual_evolution.runtime_guard import RuntimeGuardConfig, runtime_safety_decision


def sample_episode(episode_id="E-1"):
    return {
        "episode_id": episode_id, "prediction_time": "2026-10-31T10:00:00+08:00",
        "equipment_id": "PUMP-01", "operating_condition": "high_load",
        "fault_type": "bearing_fault", "task_type": "regression_and_diagnosis",
        "model": {"model_version": "M0", "activation_id": 1, "prediction": 0.4,
                  "confidence": 0.8, "confidence_calibrated": False},
        "agent": {"policy_version": "P0", "prompt_version": "PR0",
                  "tool_schema_version": "T0", "kb_version": "KB0",
                  "decision": "manual_review", "tool_calls": [], "citations": []},
    }


class TestEpisodeStore(unittest.TestCase):
    def test_review_lifecycle_and_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            store = EpisodeStore(Path(temp) / "episodes.jsonl")
            store.append_episode(sample_episode())
            with self.assertRaises(ValueError):
                store.review_episode("E-1", status="verified", reviewer="r1",
                                     source_ref="", ground_truth="bearing_fault")
            store.review_episode("E-1", status="verified", reviewer="r1",
                                 source_ref="WO-1", ground_truth="bearing_fault",
                                 failure_categories=["agent_overconfidence"],
                                 feedback_signals={"human_rating": -1, "diagnosis_overturned": True})
            item = store.get_episode("E-1")
            self.assertEqual(item["label"]["status"], "verified")
            self.assertEqual(item["failure_categories"], ["agent_overconfidence"])
            self.assertTrue(item["feedback_signals"]["diagnosis_overturned"])
            self.assertTrue(store.verify_integrity()["valid"])
            with self.assertRaises(ValueError):
                store.review_episode("E-1", status="disputed", reviewer="r2",
                                     source_ref="WO-2", ground_truth="other")

    def test_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "episodes.jsonl"
            store = EpisodeStore(path)
            store.append_episode(sample_episode())
            raw = path.read_text(encoding="utf-8").replace('"PUMP-01"', '"PUMP-02"')
            path.write_text(raw, encoding="utf-8")
            with self.assertRaises(ValueError):
                store.verify_integrity()


class TestModelGate(unittest.TestCase):
    def test_stratum_regression_rejects_global_improvement(self):
        n = 240
        y = np.zeros(n)
        base = np.full(n, 0.5)
        cand = np.full(n, 0.05)
        faults = ["normal"] * 220 + ["bearing_fault"] * 20
        base[-20:] = 0.2
        cand[-20:] = 0.4
        metadata = [{"equipment_id": "P1" if i % 2 else "P2",
                     "operating_condition": "high" if i % 3 else "normal",
                     "fault_type": faults[i]} for i in range(n)]
        result = evaluate_model_candidate(y, base, cand, metadata,
                                           ModelGateConfig(critical_fault_types=("bearing_fault",)))
        self.assertLess(result["global"]["candidate"]["mae"], result["global"]["incumbent"]["mae"])
        self.assertEqual(result["strata"]["fault_type"]["bearing_fault"]["status"], "REJECT")
        self.assertEqual(result["status"], "REJECT")
        self.assertFalse(result["allow_upgrade"])

    def test_insufficient_critical_samples_requires_review(self):
        n = 30
        y = np.zeros(n)
        pred = np.ones(n) * 0.1
        metadata = [{"equipment_id": "P1", "operating_condition": "normal",
                     "fault_type": "rare_fault" if i < 3 else "normal"} for i in range(n)]
        result = evaluate_model_candidate(y, pred, pred, metadata,
                                           ModelGateConfig(critical_fault_types=("rare_fault",)))
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertFalse(result["allow_upgrade"])


class TestAgentAndSafety(unittest.TestCase):
    def test_agent_gate_rejects_missing_citation(self):
        cases = [{"case_id": "C1", "allowed_decisions": ["manual_review"],
                  "required_tools": ["retrieve_manual"], "citations_required": True,
                  "manual_review_expected": True, "critical": True}]
        outputs = [{"decision": "manual_review", "manual_review_requested": True,
                    "tool_calls": [{"name": "retrieve_manual", "success": True}],
                    "citations": []}]
        result = evaluate_agent_candidate(cases, outputs)
        self.assertEqual(result["status"], "REJECT")
        self.assertEqual(result["metrics"]["critical_failures"], 1)

    def test_uncalibrated_confidence_forces_human_review(self):
        cfg = RuntimeGuardConfig(calibration_id="cal-v1")
        result = runtime_safety_decision(ood_score=0.1, drift_score=0.05,
                                         confidence=0.99, confidence_calibrated=False,
                                         config=cfg)
        self.assertEqual(result["mode"], "human_review")
        self.assertFalse(result["allow_automatic_control"])

    def test_normal_requires_calibration(self):
        cfg = RuntimeGuardConfig(calibration_id="cal-v1")
        result = runtime_safety_decision(ood_score=0.1, drift_score=0.05,
                                         confidence=0.95, confidence_calibrated=True,
                                         config=cfg)
        self.assertEqual(result["mode"], "normal")


class TestKnowledgeAndRegistry(unittest.TestCase):
    def test_only_approved_effective_knowledge_retrieves(self):
        with tempfile.TemporaryDirectory() as temp:
            kb = KnowledgeCatalog(Path(temp) / "kb.jsonl")
            doc = {"document_id": "MAN-1", "version": "1.0", "title": "泵轴承维护手册",
                   "source_uri": "manual://pump/1", "equipment_scope": ["PUMP-01"],
                   "effective_from": "2026-01-01", "effective_to": None,
                   "summary": "轴承振动温升检查", "content": "泵设备轴承检查和振动监测。",
                   "keywords": ["泵", "轴承"]}
            kb.register_document(doc)
            self.assertEqual(kb.search("泵轴承振动", equipment_id="PUMP-01"), [])
            kb.approve_document("MAN-1", "1.0", reviewer="expert", approval_ref="APP-1")
            self.assertTrue(kb.search("泵轴承振动", equipment_id="PUMP-01"))
            self.assertEqual(kb.search("泵轴承", equipment_id="PUMP-02"), [])
            self.assertTrue(kb.verify_integrity()["valid"])

    def test_joint_release_requires_all_gates(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = JointReleaseRegistry(temp)
            manifest = {"release_id": "R1", "model_version": "M1", "model_activation_id": 1,
                        "model_checkpoint_sha256": "b" * 64, "agent_policy_version": "P1",
                        "prompt_version": "PR1", "tool_schema_version": "T1", "kb_version": "K1",
                        "retrieval_config_version": "RET1", "risk_policy_version": "SAFE1",
                        "code_version": "git-sha",
                        "gate_evidence": {"model_gate_ref": "gates/model.json",
                                          "agent_gate_ref": "gates/agent.json",
                                          "shadow_ref": "shadow/run-001"}}
            registry.register_release(manifest)
            with self.assertRaises(ValueError):
                registry.activate("R1", approver="owner", model_gate_status="REJECT",
                                  agent_gate_status="PASS", shadow_status="PASS",
                                  safety_policy_approved=True, approval_ref="APP")
            pointer = registry.activate("R1", approver="owner", model_gate_status="PASS",
                                        agent_gate_status="PASS", shadow_status="PASS",
                                        safety_policy_approved=True, approval_ref="APP")
            self.assertEqual(pointer["release_id"], "R1")
            self.assertEqual(registry.current()["release_id"], "R1")
            # Simulate interruption after event fsync but before pointer replacement.
            registry.current_path.unlink()
            self.assertEqual(registry.current()["release_id"], "R1")
            self.assertTrue(registry.verify_integrity()["valid"])

    def test_rollback_only_targets_a_previously_activated_release(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = JointReleaseRegistry(temp)
            def manifest(release_id, model_version):
                return {"release_id": release_id, "model_version": model_version,
                        "model_activation_id": 1, "model_checkpoint_sha256": "c" * 64,
                        "agent_policy_version": "P1", "prompt_version": "PR1",
                        "tool_schema_version": "T1", "kb_version": "K1",
                        "retrieval_config_version": "RET1", "risk_policy_version": "SAFE1",
                        "code_version": "git-sha",
                        "gate_evidence": {"model_gate_ref": "mg", "agent_gate_ref": "ag", "shadow_ref": "sh"}}
            registry.register_release(manifest("R0", "M0"))
            registry.activate("R0", approver="owner", model_gate_status="PASS",
                              agent_gate_status="PASS", shadow_status="PASS",
                              safety_policy_approved=True, approval_ref="APP0")
            registry.register_release(manifest("R1", "M1"))
            with self.assertRaises(ValueError):
                registry.rollback("R1", approver="owner", approval_ref="APP", reason="test")
            pointer = registry.rollback("R0", approver="owner", approval_ref="APP", reason="rollback test")
            self.assertEqual(pointer["action"], "rollback")
            self.assertEqual(registry.current()["release_id"], "R0")


class TestAgentCompare(unittest.TestCase):
    def test_candidate_agent_may_not_break_a_previously_passing_case(self):
        cases = [{"case_id": "C1", "allowed_decisions": ["manual_review"],
                  "manual_review_expected": True, "critical": True}]
        incumbent = [{"decision": "manual_review", "manual_review_requested": True,
                      "tool_calls": [], "citations": []}]
        candidate = [{"decision": "automatic_diagnosis", "manual_review_requested": False,
                      "tool_calls": [], "citations": []}]
        result = compare_agent_candidates(cases, incumbent, candidate)
        self.assertEqual(result["status"], "REJECT")
        self.assertEqual(len(result["new_regressions"]), 1)


class TestDemo(unittest.TestCase):
    def test_demo_artifacts_and_expected_guardrails(self):
        with tempfile.TemporaryDirectory() as temp:
            report = run_demo(Path(temp))
            self.assertEqual(report["model_gate"]["status"], "REJECT")
            self.assertEqual(report["agent_gate"]["status"], "PASS")
            self.assertEqual(report["runtime_guard"]["mode"], "human_review")
            self.assertEqual(report["knowledge"]["draft_search_hits"], 0)
            self.assertGreater(report["knowledge"]["approved_search_hits"], 0)
            self.assertIsNone(report["joint_registry"]["current"])
            self.assertTrue((Path(temp) / "demo_report.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
