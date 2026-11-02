import tempfile
import unittest
from pathlib import Path

from diagnostic_agent.agent import AgentRuntimeConfig, DiagnosticAgent
from diagnostic_agent.providers import OfflinePlanner
from dual_evolution.episode_store import EpisodeStore
from dual_evolution.knowledge_catalog import KnowledgeCatalog
from dual_evolution.runtime_guard import RuntimeGuardConfig


class StrategyCandidateGenerationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = EpisodeStore(self.root / "episodes.jsonl")
        self.agent = DiagnosticAgent(
            planner=OfflinePlanner(), knowledge=KnowledgeCatalog(self.root / "kb.jsonl"),
            episode_store=self.store, workspace=self.root,
            config=AgentRuntimeConfig(strategy_min_support=2,
                runtime_guard=RuntimeGuardConfig(calibration_id="TEST-CAL")))

    def tearDown(self):
        self.tmp.cleanup()

    def _create_episode(self, episode_id):
        self.agent.run({
            "episode_id": episode_id, "prediction_time": "2026-11-02T08:00:00+08:00",
            "equipment_id": "PUMP-STRATEGY-01", "operating_condition": "steady_load",
            "sensor_readings": {"bearing_temp_c": [60.0, 61.0]},
            "model": {"model_name": "LSTM", "model_version": "M1", "activation_id": 1},
            "model_prediction": {"fault_type": "bearing_fault", "confidence": 0.9,
                "confidence_calibrated": True, "ood_score": 0.1, "drift_score": 0.05},
        })

    def _review(self, episode_id, category):
        return self.agent.submit_feedback(
            episode_id, reviewer="maintenance-engineer", source_ref=f"workorder://{episode_id}",
            status="verified", ground_truth={"fault_type": "bearing_fault"},
            corrected_decision="人工确认的处置结论", failure_categories=[category],
            note="经维修工单核实，Agent 原处理流程需要调整。")

    def test_verified_corrections_auto_generate_prompt_route_and_retrieval_candidates(self):
        categories = ["agent_overconfidence", "tool_routing_error", "retrieval_miss"]
        results = []
        counter = 0
        for category in categories:
            for _ in range(2):
                counter += 1
                episode_id = f"CAND-{counter:03d}"
                self._create_episode(episode_id)
                results.append(self._review(episode_id, category))
        all_candidates = self.agent.strategy_candidates.list_candidates()
        types = {x["candidate_type"] for x in all_candidates}
        self.assertEqual(types, {"prompt", "tool_route", "retrieval"})
        self.assertTrue(all(x["requires_human_approval"] for x in all_candidates))
        self.assertTrue(all(x["requires_frozen_replay"] for x in all_candidates))
        self.assertTrue(all(x["applied_to_live_agent"] is False for x in all_candidates))
        self.assertEqual(len(all_candidates), 3)

    def test_single_verified_correction_does_not_create_candidate(self):
        self._create_episode("SINGLE-001")
        result = self._review("SINGLE-001", "retrieval_miss")
        self.assertEqual(result["strategy_candidates_generated"], 0)

    def test_disputed_feedback_does_not_generate_candidate(self):
        self._create_episode("DISPUTED-001")
        result = self.agent.submit_feedback(
            "DISPUTED-001", reviewer="maintenance-engineer", source_ref="workorder://disputed",
            status="disputed", failure_categories=["retrieval_miss"], note="待进一步核实")
        self.assertEqual(result["strategy_candidates_generated"], 0)

    def test_candidate_generation_is_idempotent_for_same_evidence(self):
        for episode_id in ("IDEMP-001", "IDEMP-002"):
            self._create_episode(episode_id)
            result = self._review(episode_id, "retrieval_miss")
        self.assertEqual(result["strategy_candidates_generated"], 1)
        proposals = self.agent.strategy_candidates.generate(
            self.store.list_episodes(status="verified"),
            base_versions={"prompt_version": self.agent.config.prompt_version,
                           "tool_schema_version": self.agent.config.tool_schema_version,
                           "retrieval_config_version": self.agent.config.retrieval_config_version,
                           "policy_version": self.agent.config.policy_version})
        self.assertEqual(proposals, [])


if __name__ == "__main__":
    unittest.main()
