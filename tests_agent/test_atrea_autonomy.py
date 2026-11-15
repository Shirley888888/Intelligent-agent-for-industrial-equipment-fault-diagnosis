import json
import tempfile
import unittest
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from atrea.store import ExperienceStore
from atrea.scheduler import next_run, run_due_once
from atrea.ablation import run_three_arm_ablation
from atrea.cycle import run_research_cycle
from atrea.chronos2_runner import Chronos2ExperimentRunner


class AtreaAutonomyTests(unittest.TestCase):
    def test_experience_roundtrip_and_scope_isolation(self):
        with tempfile.TemporaryDirectory() as d:
            s = ExperienceStore(Path(d) / "x.sqlite3")
            base = {"batch_id":"b", "dataset":"ETTh1", "split_hash":"s",
                    "model_name":"LSTM", "config":{}, "hyperparameters":{}, "metrics":{"mae":1.2}}
            s.add_experiment({**base, "memory_scope":"A3"})
            s.add_experiment({**base, "batch_id":"b2", "model_name":"TCN", "memory_scope":"A2"})
            self.assertEqual(len(s.retrieve_experience(dataset="ETTh1", memory_scope="A3")["experiments"]), 1)
            self.assertEqual(len(s.retrieve_experience(dataset="ETTh1", memory_scope="A2")["experiments"]), 1)
            self.assertEqual(len(s.retrieve_experience(dataset="ETTh1", memory_scope="default")["experiments"]), 0)

    def test_next_run_china_timezone(self):
        now = datetime(2026, 11, 5, 7, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        self.assertEqual(next_run(now).hour, 8)
        self.assertEqual(next_run(now).date(), now.date())

    def test_scheduler_failure_is_retryable_and_success_is_once_daily(self):
        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / "state.json"
            now = datetime(2026, 11, 6, 8, 5, tzinfo=ZoneInfo("Asia/Shanghai"))
            failed = run_due_once(lambda **kw: (_ for _ in ()).throw(RuntimeError("network")),
                                  state_path=state, now=now, retry_seconds=0)
            self.assertEqual(failed["status"], "failed")
            succeeded = run_due_once(lambda **kw: {"ok": True}, state_path=state,
                                     now=now, retry_seconds=0)
            self.assertEqual(succeeded["status"], "success")
            again = run_due_once(lambda **kw: self.fail("must not run twice"),
                                 state_path=state, now=now, retry_seconds=0)
            self.assertEqual(again["status"], "already_succeeded")

    def test_three_arm_ablation_keeps_a2_out_of_experience_memory(self):
        with tempfile.TemporaryDirectory() as d:
            s = ExperienceStore(Path(d) / "x.sqlite3")
            def search(q, max_results=20):
                return [{"paper_id":"p1", "title":"A paper", "url":"https://example.org/p", "abstract":"method"}]
            a3_runs = {"count": 0}
            def runner(arm, papers, memory, batch, experience_tool=None):
                self.assertEqual(memory["experiments"], [], "no historical records may be auto-injected")
                considered = False
                if arm["use_memory"]:
                    result = experience_tool.search_experience("ETTh1 forecasting", dataset="ETTh1")
                    considered = bool(result["results"]) or experience_tool.calls > 0
                    a3_runs["count"] += 1
                return [{"dataset":"ETTh1", "split_hash":"same", "model_name":arm["id"],
                    "config":{}, "hyperparameters":{"seed":1}, "seed":1,
                    "metrics":{"mae":1.0}, "architecture_changed":arm["change_architecture"],
                    "experience_considered":considered}]
            r = run_three_arm_ablation(store=s, experiment_runner=runner,
                paper_searcher=search, output_dir=Path(d)/"out")
            self.assertEqual(len(r["results"]), 3)
            self.assertTrue(Path(r["summary_csv"]).exists())
            # Only A3 is persisted to reusable memory; A1/A2 artifacts are still saved separately.
            self.assertEqual(len(s.retrieve_experience(memory_scope="A3_experience_enhanced")["experiments"]), 1)
            self.assertEqual(len(s.retrieve_experience(memory_scope="A2_online_no_memory")["experiments"]), 0)
            self.assertTrue(list((Path(d)/"out").glob("A2_online_no_memory_*.json")))
            # Second pass proves A3 reads its own previous result while A2 remains memory-free.
            run_three_arm_ablation(store=s, experiment_runner=runner,
                paper_searcher=search, output_dir=Path(d)/"out")
            self.assertEqual(a3_runs["count"], 2)
            self.assertEqual(len(s.recent_papers()), 1)

    def test_chronos2_runner_rejects_unreviewed_config_keys(self):
        with tempfile.TemporaryDirectory() as d:
            script = Path(d) / "train_chronos2_ett.py"
            script.write_text("# fixture", encoding="utf-8")
            with self.assertRaises(ValueError):
                Chronos2ExperimentRunner(d, base_config={"arbitrary_python": "not allowed"})
            runner = Chronos2ExperimentRunner(d, base_config={"frequency":"hourly"})
            with self.assertRaises(ValueError):
                runner({"plan":{"chronos2_config":{"out_dir":"../../unsafe"}}}, "batch")

    def test_cycle_residual_diagnostics_and_persisted_report(self):
        with tempfile.TemporaryDirectory() as d:
            s = ExperienceStore(Path(d) / "x.sqlite3")
            def search(q, max_results=15, submitted_after=None):
                return [{"paper_id":"fixture", "title":"Fixture paper", "url":"https://example.org/p",
                         "abstract":"patch forecasting", "published":"2026-01-01",
                         "source":"test", "extracted_methods":["patching"]}]
            def runner(proposal, batch):
                return [{"dataset":"ETTh1", "split_hash":"split-v1", "model_name":"candidate",
                         "config":{}, "hyperparameters":{"seed":1}, "seed":1,
                         "metrics":{"mae":0.5}, "is_candidate":True,
                         "y_true":[1,2,3], "y_pred":[1.1,1.8,3.2]}]
            result = run_research_cycle(store=s, train_and_evaluate=runner, paper_search=search,
                hypothesis_builder=lambda papers,memory:{"statement":"Does patching help?",
                    "rationale":"testable", "plan":{"models":["candidate"],"same_split":True}},
                memory_scope="A3")
            self.assertEqual(result["experience_used"]["experiments"], 0)
            self.assertTrue(Path(result["report_path"]).exists())
            self.assertTrue(Path(result["trace_path"]).exists())
            trace_rows = [json.loads(x) for x in Path(result["trace_path"]).read_text(encoding="utf-8").splitlines()]
            self.assertIn("form_falsifiable_hypothesis", [x["action"] for x in trace_rows])
            metrics = s.batch_experiments(result["batch_id"])[0]["metrics"]
            self.assertAlmostEqual(metrics["error_diagnostics"]["mae"], 0.1666666667, places=6)


if __name__ == "__main__":
    unittest.main()
