import json
import tempfile
import unittest
from pathlib import Path

from diagnostic_agent.agent import AgentRuntimeConfig, DiagnosticAgent
from diagnostic_agent.providers import OfflinePlanner
from dual_evolution.episode_store import EpisodeStore
from dual_evolution.agent_gate import evaluate_agent_candidate
from dual_evolution.knowledge_catalog import KnowledgeCatalog
from dual_evolution.runtime_guard import RuntimeGuardConfig


def context(**overrides):
    value = {
        "episode_id": "TEST-CORE-001", "prediction_time": "2026-11-02T08:00:00+08:00",
        "equipment_id": "PUMP-T1", "operating_condition": "steady_load",
        "sensor_readings": {"bearing_temp_c": [65.0, 66.0], "vibration_mm_s": [2.0, 2.2]},
        "model": {"model_name": "TCN", "model_version": "M1", "activation_id": 3},
        "model_prediction": {"fault_type": "bearing_fault", "confidence": 0.91,
            "confidence_calibrated": True, "ood_score": 0.1, "drift_score": 0.05,
            "anomaly_score": 0.8, "rul_hours": 48.0, "fault_amplitude": 0.33},
    }
    value.update(overrides)
    return value


def make_agent(root, planner=None):
    knowledge = KnowledgeCatalog(root / "knowledge.jsonl")
    knowledge.register_document({
        "document_id": "DOC-PUMP-01", "version": "1.0", "title": "泵轴承维护知识（测试夹具）",
        "source_uri": "test://manual/pump/1", "equipment_scope": ["PUMP-T1"],
        "effective_from": "2026-01-01", "effective_to": None,
        "summary": "泵 轴承 温度 振动 检查", "content": "测试夹具：泵轴承温度与振动检查，仅验证引用链路，不作为维修规程。",
        "keywords": ["泵", "轴承", "温度", "振动"], "registered_by": "test"})
    knowledge.approve_document("DOC-PUMP-01", "1.0", reviewer="reviewer", approval_ref="test://approval/1")
    cfg = AgentRuntimeConfig(runtime_guard=RuntimeGuardConfig(calibration_id="TEST-CAL-1"))
    return DiagnosticAgent(planner=planner or OfflinePlanner(), knowledge=knowledge,
                           episode_store=EpisodeStore(root / "episodes.jsonl"), workspace=root,
                           config=cfg)


class Iteration1CoreTests(unittest.TestCase):
    def test_end_to_end_offline_run_creates_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent = make_agent(root)
            result = agent.run(context())
            self.assertEqual(result["diagnosis"]["fault_type"], "bearing_fault")
            self.assertEqual(result["diagnosis"]["fault_type_source"], "independent_time_series_model")
            self.assertFalse(result["diagnosis"]["fault_type_overridable_by_agent"])
            self.assertTrue(result["episode_persistence"]["stored"])
            self.assertEqual(len(result["tool_calls"]), 2)
            self.assertGreaterEqual(len(result["citations"]), 1)
            self.assertEqual(agent.episode_store.get_episode("TEST-CORE-001")["label"]["status"], "pending")
            replay_case = [{"case_id": "ACTUAL-AGENT-OUTPUT", "allowed_decisions": ["provisional_model_result"],
                            "manual_review_expected": False,
                            "required_tools": ["search_knowledge", "summarize_sensor_window"],
                            "citations_required": True, "critical": True}]
            gate = evaluate_agent_candidate(replay_case, [result], min_case_pass_rate=1.0,
                min_tool_success_rate=1.0, min_citation_coverage=1.0, max_critical_failures=0)
            self.assertEqual(gate["status"], "PASS")

    def test_model_label_is_not_overridden_by_llm(self):
        class HallucinatingPlanner:
            def complete(self, messages, tools):
                if not any(x.get("role") == "tool" and x.get("name") == "search_knowledge" for x in messages):
                    return {"tool_calls": [{"name": "search_knowledge", "arguments": {"query": "泵 轴承", "top_k": 5}}]}
                if not any(x.get("role") == "tool" and x.get("name") == "summarize_sensor_window" for x in messages):
                    return {"tool_calls": [{"name": "summarize_sensor_window", "arguments": {}}]}
                return {"tool_calls": [], "content": json.dumps({
                    "fault_type": "pump_failure_invented", "observations": ["有必要人工核查"],
                    "suggested_checks": [], "limitations": []}, ensure_ascii=False)}
        with tempfile.TemporaryDirectory() as tmp:
            result = make_agent(Path(tmp), HallucinatingPlanner()).run(context())
            self.assertEqual(result["diagnosis"]["fault_type"], "bearing_fault")
            self.assertNotIn("fault_type", result["diagnosis"]["prediction"])

    def test_invalid_timestamp_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(Path(tmp))
            with self.assertRaises(ValueError):
                agent.run(context(prediction_time="not-a-date"))

    def test_model_provenance_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(Path(tmp))
            bad = context(model={"model_version": "M1"})
            with self.assertRaises(ValueError):
                agent.run(bad)

    def test_sensor_nan_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(Path(tmp))
            with self.assertRaises(ValueError):
                agent.run(context(sensor_readings={"bearing_temp_c": [float("nan")]}))


if __name__ == "__main__":
    unittest.main()
