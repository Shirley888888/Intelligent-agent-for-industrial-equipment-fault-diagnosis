import tempfile
import unittest
from pathlib import Path

from diagnostic_agent.tools import ToolExecutor
from dual_evolution.knowledge_catalog import KnowledgeCatalog


class Iteration2ToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.kb = KnowledgeCatalog(self.root / "kb.jsonl")
        self.tools = ToolExecutor(self.kb, self.root)
        self.context = {"equipment_id": "P-01", "prediction_time": "2026-11-02T08:00:00+08:00",
                        "sensor_readings": {"temp": [10, 20, 30], "vibration": 2.5}}

    def tearDown(self):
        self.tmp.cleanup()

    def test_unknown_tool_is_rejected_and_audited(self):
        result = self.tools.execute("shell", {"cmd": "whoami"}, self.context)
        self.assertFalse(result["success"])
        self.assertIn("not allowlisted", result["result"]["message"])
        self.assertTrue(self.tools.audit_path.exists())

    def test_unexpected_arguments_are_rejected(self):
        result = self.tools.execute("search_knowledge", {"query": "pump", "equipment_id": "OTHER"}, self.context)
        self.assertFalse(result["success"])
        self.assertIn("Unexpected", result["result"]["message"])

    def test_equipment_scope_cannot_be_overridden_by_planner(self):
        self.kb.register_document({"document_id": "DOC-A", "version": "1", "title": "泵温度",
            "source_uri": "test://a", "equipment_scope": ["P-01"], "effective_from": "2026-01-01",
            "content": "泵 温度 轴承", "summary": "泵温度", "keywords": ["泵", "温度"]})
        self.kb.approve_document("DOC-A", "1", reviewer="r", approval_ref="test://approval")
        result = self.tools.execute("search_knowledge", {"query": "泵 温度"}, self.context)
        self.assertTrue(result["success"])
        self.assertEqual(result["result"]["scope_equipment"], "P-01")
        self.assertEqual([h["document_id"] for h in result["result"]["hits"]], ["DOC-A"])

    def test_sensor_tool_only_returns_descriptive_statistics(self):
        result = self.tools.execute("summarize_sensor_window", {"metrics": ["temp"]}, self.context)
        self.assertTrue(result["success"])
        self.assertEqual(result["result"]["metrics"]["temp"]["mean"], 20.0)
        self.assertIn("no fault classification", result["result"]["note"])

    def test_sensor_tool_ignores_non_numeric_data(self):
        context = dict(self.context, sensor_readings={"temp": [1, "alert", 3]})
        result = self.tools.execute("summarize_sensor_window", {}, context)
        self.assertEqual(result["result"]["metrics"]["temp"]["count"], 2)

    def test_invalid_search_length_is_rejected(self):
        result = self.tools.execute("search_knowledge", {"query": "x" * 513}, self.context)
        self.assertFalse(result["success"])


if __name__ == "__main__":
    unittest.main()
