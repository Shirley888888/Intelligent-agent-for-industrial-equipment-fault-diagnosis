import tempfile
import unittest
from pathlib import Path

from atrea.store import ExperienceStore
from atrea.experience import SearchExperienceTool
from atrea.research import build_research_hypothesis
from atrea.cycle import run_research_cycle


class ExperiencePaperTests(unittest.TestCase):
    def test_paper_archive_search_and_live_catalog(self):
        with tempfile.TemporaryDirectory() as td:
            store = ExperienceStore(Path(td) / "agent.sqlite3")
            saved = store.add_experience_paper(
                title="ETTh1 长预测跨度误差分析与分解方法研究",
                topic="time series forecasting", content="## 结果\nPatchTST 与趋势分解方法。",
                dataset="ETTh1", tags=["PatchTST", "decomposition"], memory_scope="test")
            catalog = Path(saved["catalog_path"])
            self.assertTrue(Path(saved["file_path"]).exists())
            self.assertIn("ETTh1 长预测跨度误差分析与分解方法研究", catalog.read_text(encoding="utf-8"))
            tool = SearchExperienceTool(store, memory_scope="test")
            result = tool.search_experience("ETTh1 decomposition", dataset="ETTh1")
            self.assertEqual(len(result["results"]), 1)
            self.assertEqual(result["results"][0]["experience_id"], saved["experience_id"])

    def test_cycle_does_not_inject_prior_experience_and_archives_new_paper(self):
        with tempfile.TemporaryDirectory() as td:
            store = ExperienceStore(Path(td) / "agent.sqlite3")
            store.add_experience_paper(title="Prior ETTh1 paper", topic="ETTh1",
                content="PatchTST patching", dataset="ETTh1")
            def paper_search(query, max_results=15, submitted_after=None):
                return [{"paper_id":"p1", "title":"One external paper", "url":"https://example.org/p1",
                         "abstract":"few methods", "extracted_methods":[]}]
            def planner(papers, memory, experience_tool):
                self.assertEqual(memory, {"experiments": [], "patterns": []})
                found = experience_tool.search_experience("ETTh1 PatchTST", dataset="ETTh1")
                self.assertTrue(found["results"])
                return {"statement":"Test a paper-derived method", "rationale":"test",
                        "plan":{"dataset":"ETTh1", "models":["baseline"]}}
            def runner(proposal, batch_id):
                return [{"dataset":"ETTh1", "split_hash":"frozen", "model_name":"baseline",
                         "config":{}, "hyperparameters":{}, "metrics":{"mae":0.4}}]
            result = run_research_cycle(store=store, train_and_evaluate=runner,
                paper_search=paper_search, hypothesis_builder=planner)
            self.assertEqual(result["experience_used"]["experiments"], 0)
            self.assertEqual(result["experience_used"]["search_tool_calls"], 1)
            self.assertTrue(Path(result["experience_paper"]["file_path"]).exists())
            catalog = Path(result["experience_paper"]["catalog_path"]).read_text(encoding="utf-8")
            self.assertIn(result["experience_paper"]["title"], catalog)

    def test_default_planner_selectively_calls_search_tool(self):
        class FakeTool:
            def __init__(self): self.calls = []
            def search_experience(self, query, **kwargs):
                self.calls.append(query)
                return {"results": [{"experience_id": "e1", "title": "PatchTST patching",
                    "content": "PatchTST patch-based forecasting method.", "file_path": "paper.md", "score": 2}]}
        tool = FakeTool()
        proposal = build_research_hypothesis([], {"experiments": [], "patterns": []},
                                             experience_tool=tool, query="ETTh1 forecasting")
        self.assertEqual(len(tool.calls), 1)
        self.assertTrue(proposal["experience_paper_evidence"])
        self.assertIn("patching", proposal["plan"]["candidate_methods"])

        tool2 = FakeTool()
        papers = [{"title": f"Forecast paper {i}", "abstract": "PatchTST patch-based method",
                   "extracted_methods": ["patching"]} for i in range(4)]
        build_research_hypothesis(papers, {"experiments": [], "patterns": []},
                                  experience_tool=tool2, query="ETTh1")
        self.assertEqual(tool2.calls, [])


if __name__ == "__main__":
    unittest.main()
