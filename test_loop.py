"""Focused invariants; unittest only. Results are engineering tests, not model gains."""
import json
import tempfile
import unittest
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

from minimal_loop import (FeedbackMonitor, ForecastModel, VersionManager, complete_feedback,
                          load_csv, model_gate, simulate, predict_tool, train_candidate_tool,
                          make_version, TrainConfig, GateConfig, FEATURES, L, H)


class LoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.times = load_csv(Path(__file__).with_name("ETTh1.csv"))
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.a = cls.root/"original"
        cls.summary = simulate(cls.data, cls.times, cls.a, exercise_at=60, max_episodes=61)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_future_perturbation_does_not_change_prefix(self):
        cutoff = self.summary["observed_end"]
        changed = self.data.copy()
        changed[cutoff:] += 10000.0
        b = self.root/"changed_future"
        summary = simulate(changed, self.times, b, exercise_at=60, max_episodes=61)
        self.assertEqual(self.summary, summary)
        for name in ["predictions.jsonl", "feedback.jsonl", "monitor.jsonl"]:
            self.assertEqual((self.a/name).read_text(), (b/name).read_text())
        for candidate in ["C0", "C1"]:
            with np.load(self.a/"candidates"/candidate/"checkpoint.npz") as x, \
                    np.load(b/"candidates"/candidate/"checkpoint.npz") as y:
                for key in ["w", "mean", "scale", "metadata"]:
                    np.testing.assert_array_equal(x[key], y[key])

    def test_feedback_alignment_and_delayed_release(self):
        episodes = [json.loads(line) for line in (self.a/"feedback.jsonl").read_text().splitlines()]
        for e in episodes:
            p, end = e["origin_idx"], e["target_end_idx"]
            self.assertEqual(end-p, 24)
            self.assertLess(e["history_times"][-1], e["prediction_time"])
            self.assertEqual(e["target_times"][0], e["prediction_time"])
            self.assertLess(e["target_times"][-1], e["feedback_time"])
            np.testing.assert_array_equal(e["truth"], self.data[p:end, 6])
            self.assertAlmostEqual(e["MAE"], np.abs(np.array(e["forecast"])-e["truth"]).mean())
        pending = json.loads((self.a/"predictions.jsonl").read_text().splitlines()[0])
        with self.assertRaises(ValueError):
            complete_feedback(pending, self.data[:pending["target_end_idx"]-1],
                              self.times[:pending["target_end_idx"]-1], datetime.fromisoformat(pending["prediction_time"])+timedelta(hours=23))

    def test_monitor_two_blocks_and_version_filter(self):
        template = json.loads((self.a/"feedback.jsonl").read_text().splitlines()[0])
        monitor = FeedbackMonitor(n=2, patience=2, cooldown_episodes=2)
        episodes = []
        for i in range(4):
            e = deepcopy(template)
            e.update(episode_id=str(i), origin_idx=i*24, target_end_idx=(i+1)*24, MAE=2.0)
            episodes.append(e)
            decision = monitor.run(episodes, "M0", 0, 1.0, "2099-01-01T00:00:00")
            self.assertEqual(decision["trigger_retrain"], i == 3)
        monitor.after_attempt(4)
        monitor.reset()
        decision = monitor.run(episodes, "M1", 1, 1.0, "2099-01-01T00:00:00")
        self.assertEqual(decision["count"], 0)
        self.assertFalse(decision["trigger_retrain"])

    def test_gate_requires_both_conditions(self):
        self.assertTrue(model_gate(1.0, 0.90, 1.0, 1.01)["allow_upgrade"])
        self.assertFalse(model_gate(1.0, 0.99, 1.0, 1.00)["allow_upgrade"])
        self.assertFalse(model_gate(1.0, 0.90, 1.0, 1.10)["allow_upgrade"])
        self.assertFalse(model_gate(0.0, 0.0, 1.0, 1.0)["allow_upgrade"])
        self.assertFalse(model_gate(1.0, float("nan"), 1.0, 1.0)["allow_upgrade"])

    def test_version_upgrade_rollback_and_corruption(self):
        # Use controlled gate observations to exercise version operations.
        # These are unit-test scores, not ETTh1 evaluation results.
        registry = VersionManager(self.root/"registry")
        candidate = json.loads((self.a/"candidates"/"C0"/"manifest.json").read_text())
        now = "2099-01-01T00:00:00"
        registry.publish(candidate, None, now, 1.0)
        # modify:gate-activation-test - fixtures bind to the actual current activation.
        def controlled_gate():
            current = registry.current()
            model, _ = registry.load(current["model_version"])
            from minimal_loop import digest
            return {**model_gate(1.0, 0.90, 1.0, 1.0),
                    "incumbent_version": current["model_version"],
                    "incumbent_activation_id": current["activation_id"],
                    "incumbent_checkpoint_sha256": digest(model.checkpoint),
                    "candidate_id": candidate["candidate_id"],
                    "checkpoint_sha256": candidate["checkpoint_sha256"],
                    "snapshot_sha256": candidate["manifest"]["snapshot_sha256"],
                    "gate_config": vars(GateConfig()),
                    "evaluated_as_of": candidate["trained_as_of"]}
        gate = controlled_gate()
        registry.publish(candidate, gate, now, 0.90)
        self.assertEqual(registry.current()["model_version"], "M1")
        registry.activate("M0", now)
        self.assertEqual(registry.current()["model_version"], "M0")
        # A gate for M0 from before this rollback is stale, even though version matches.
        before = registry.current()
        with self.assertRaises(ValueError):
            registry.publish(candidate, gate, now, 0.90)
        self.assertEqual(registry.current(), before)
        self.assertFalse((registry.root/"versions"/"M2").exists())
        registry.publish(candidate, controlled_gate(), now, 0.90)
        self.assertEqual(registry.current()["model_version"], "M2")
        registry.activate("M1", now)
        before = registry.current()
        (registry.root/"versions"/"M0"/"M0_checkpoint.npz").write_bytes(b"broken")
        with self.assertRaises(ValueError):
            registry.activate("M0", now)
        self.assertEqual(before, registry.current())

    def test_candidate_target_ranges_are_disjoint(self):
        m = json.loads((self.a/"candidates"/"C1"/"manifest.json").read_text())["manifest"]
        vs, ve = m["validation_range"]
        hs, he = m["historical_range"]
        rs, re = m["recent_range"]
        self.assertLessEqual(he, vs)
        self.assertEqual(ve, rs)
        self.assertLessEqual(re, m["observed_end"])
        for start, end in m["train_ranges"]:
            self.assertLessEqual(end, vs)
            self.assertTrue(end <= hs or start >= he)

    # modify:revision-tests - externally visible contracts and failure cases.
    def test_runtime_checks_survive_python_optimization(self):
        script = '''
import numpy as np
from datetime import datetime, timedelta
from minimal_loop import check_observed
clock = datetime(2020, 1, 1)
try:
    check_observed(np.zeros((1,7)), [clock], clock)
except ValueError:
    print("future rejected")
else:
    raise RuntimeError("future validation disabled")
'''
        result = subprocess.run([sys.executable, "-O", "-c", script],
                                cwd=Path(__file__).parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("future rejected", result.stdout)

    def test_rolling_monitor_and_batched_feedback(self):
        template = json.loads((self.a/"feedback.jsonl").read_text().splitlines()[0])
        episodes = []
        for i in range(5):
            e = deepcopy(template)
            e.update(episode_id=f"batch-{i}", origin_idx=i*24, target_end_idx=(i+1)*24, MAE=2.0)
            episodes.append(e)
        monitor = FeedbackMonitor(n=2, patience=2)
        decision = monitor.run(episodes, "M0", 0, 1.0, "2099-01-01T00:00:00")
        self.assertEqual(decision["evaluated_blocks"], 2)
        self.assertEqual(decision["recent_mae"], 2.0)
        self.assertTrue(decision["trigger_retrain"])
        monitor.after_attempt(5)
        again = monitor.run(episodes, "M0", 0, 1.0, "2099-01-01T00:00:00")
        self.assertEqual(again["recent_mae"], 2.0)
        self.assertEqual(again["evaluated_blocks"], 0)
        self.assertFalse(again["trigger_retrain"])
        reset = monitor.run(episodes, "M0", 1, 1.0, "2099-01-01T00:00:00")
        self.assertEqual(reset["count"], 0)

    def test_invalid_feedback_never_becomes_a_score(self):
        ep = json.loads((self.a/"predictions.jsonl").read_text().splitlines()[0])
        end = ep["target_end_idx"]
        now = self.times[end-1]+timedelta(hours=1)
        for field, value in [("forecast", [0.0]), ("target_end_idx", end+1),
                             ("prediction_time", "1999-01-01T00:00:00")]:
            bad = deepcopy(ep)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                complete_feedback(bad, self.data[:end], self.times[:end], now)
        leaked_times = self.times[:end].copy()
        leaked_times[0] = now
        with self.assertRaises(ValueError):
            complete_feedback(ep, self.data[:end], leaked_times, now)

    def test_invalid_ranges_and_candidate_config(self):
        from minimal_loop import DataVersion, check_manifest, validate_candidate
        candidate = json.loads((self.a/"candidates"/"C1"/"manifest.json").read_text())
        m = deepcopy(candidate["manifest"])
        m["train_ranges"].append(m["train_ranges"][0])
        end = m["observed_end"]
        with self.assertRaises(ValueError):
            check_manifest(DataVersion(**m), self.data[:end], self.times[:end])
        bad = deepcopy(candidate)
        bad["train_config"]["ridge"] = 10.0
        with self.assertRaises(ValueError):
            validate_candidate(bad)
        with self.assertRaises(ValueError):
            TrainConfig(batch_size=0).validate()
        with self.assertRaises(ValueError):
            GateConfig(old_tolerance=-1).validate()

    def test_rollback_clock_and_manifest_are_checked(self):
        candidate = json.loads((self.a/"candidates"/"C0"/"manifest.json").read_text())
        registry = VersionManager(self.root/"clock_registry")
        registry.publish(candidate, None, "2099-01-01T00:00:00", 1.0)
        before = registry.current()
        with self.assertRaises(ValueError):
            registry.activate("M0", "2098-01-01T00:00:00")
        manifest_path = registry.root/"versions"/"M0"/"manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["train_config"]["ridge"] = 2.0
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            registry.activate("M0", "2099-01-01T00:00:00")
        self.assertEqual(registry.current(), before)

    def test_catalog_and_custom_configuration(self):
        from tool_catalog import tool_catalog
        catalog = tool_catalog()
        names = {t["name"] for t in catalog["tools"]}
        self.assertTrue({"Prediction", "Train Candidate", "Model Gate", "Rollback"} <= names)
        self.assertEqual(set(catalog["structures"]["FeedbackEpisode"]["required"]),
                         set(json.loads((self.a/"feedback.jsonl").read_text().splitlines()[0])))
        json.dumps(model_gate(float("nan"), 0.9, 1.0, 1.0), allow_nan=False)
        result = simulate(self.data, self.times, self.root/"custom", max_episodes=1,
                          monitor_relative=0.3, monitor_absolute=0.2,
                          gate_config=GateConfig(recent_gain=0.10))
        settings = json.loads((self.root/"custom"/"run_config.json").read_text())
        self.assertEqual(settings["monitor"]["relative"], 0.3)
        self.assertEqual(settings["gate_config"]["recent_gain"], 0.10)
        self.assertEqual(result["complete_episodes"], 1)

    def test_failed_candidate_keeps_active_model(self):
        from unittest.mock import patch
        import minimal_loop
        actual = minimal_loop.train_candidate_tool
        def fail_after_bootstrap(*args, **kwargs):
            if kwargs.get("bootstrap", False):
                return actual(*args, **kwargs)
            raise RuntimeError("controlled training failure")
        with patch.object(minimal_loop, "train_candidate_tool", side_effect=fail_after_bootstrap):
            summary = simulate(self.data, self.times, self.root/"failed_candidate",
                               max_episodes=31, exercise_at=30)
        self.assertEqual(summary["current"]["model_version"], "M0")
        self.assertEqual(summary["upgrades"], 0)
        self.assertEqual(len(summary["candidate_errors"]), 1)
        self.assertIn("controlled training failure", summary["candidate_errors"][0])
        predictions = [json.loads(s) for s in (self.root/"failed_candidate"/"predictions.jsonl").read_text().splitlines()]
        self.assertTrue(all(e["model_version"] == "M0" for e in predictions))

    # modify:activation-fault-tests - controlled scores test state, not model gains.
    def activation_fixture(self, bootstrap=True):
        from minimal_loop import digest
        registry = VersionManager(self.root/self._testMethodName)
        candidate = json.loads((self.a/"candidates"/"C0"/"manifest.json").read_text())
        now = "2099-01-01T00:00:00"
        if bootstrap:
            registry.publish(candidate, None, now, 1.0)
        def gate():
            current = registry.current()
            model, _ = registry.load(current["model_version"])
            return {**model_gate(1.0, 0.90, 1.0, 1.0),
                    "incumbent_version": current["model_version"],
                    "incumbent_activation_id": current["activation_id"],
                    "incumbent_checkpoint_sha256": digest(model.checkpoint),
                    "candidate_id": candidate["candidate_id"],
                    "checkpoint_sha256": candidate["checkpoint_sha256"],
                    "snapshot_sha256": candidate["manifest"]["snapshot_sha256"],
                    "gate_config": vars(GateConfig()),
                    "evaluated_as_of": candidate["trained_as_of"]}
        return registry, candidate, now, gate

    def test_upgrade_event_failure_preserves_pointer_and_allows_retry(self):
        registry, candidate, now, gate = self.activation_fixture()
        before = registry.current()
        log = (registry.root/"version_events.jsonl").read_bytes()
        with patch("minimal_loop.append_json", side_effect=OSError("controlled event failure")):
            with self.assertRaisesRegex(OSError, "controlled event failure"):
                registry.publish(candidate, gate(), now, 0.90)
        self.assertEqual(registry.current(), before)
        self.assertEqual((registry.root/"version_events.jsonl").read_bytes(), log)
        self.assertFalse((registry.root/"versions"/"M1").exists())
        upgraded = registry.publish(candidate, gate(), now, 0.90)
        self.assertEqual(upgraded["model_version"], "M1")
        self.assertEqual(upgraded["activation_id"], 1)
        records = (registry.root/"version_events.jsonl").read_text().splitlines()
        self.assertEqual(len(records), 2)
        self.assertEqual(json.loads(records[-1])["previous"], before)

    def test_rollback_event_failure_preserves_pointer(self):
        registry, candidate, now, gate = self.activation_fixture()
        registry.publish(candidate, gate(), now, 0.90)
        before = registry.current()
        log = (registry.root/"version_events.jsonl").read_bytes()
        with patch("minimal_loop.append_json", side_effect=OSError("rollback event failure")):
            with self.assertRaisesRegex(OSError, "rollback event failure"):
                registry.activate("M0", now)
        self.assertEqual(registry.current(), before)
        self.assertEqual((registry.root/"version_events.jsonl").read_bytes(), log)
        rolled = registry.activate("M0", now)
        self.assertEqual(rolled["activation_id"], 2)
        self.assertEqual(rolled["model_version"], "M0")

    def test_partial_event_failure_restores_exact_journal_bytes(self):
        registry, candidate, now, gate = self.activation_fixture()
        before = registry.current()
        journal = registry.root/"version_events.jsonl"
        log = journal.read_bytes()
        def partial_append(path, obj):
            with path.open("ab") as f:
                f.write(b'{"incomplete":')
            raise OSError("partial write failure")
        with patch("minimal_loop.append_json", side_effect=partial_append):
            with self.assertRaisesRegex(OSError, "partial write failure"):
                registry.publish(candidate, gate(), now, 0.90)
        self.assertEqual(registry.current(), before)
        self.assertEqual(journal.read_bytes(), log)
        self.assertFalse((registry.root/"versions"/"M1").exists())

    def test_pointer_replace_failure_restores_event_and_cleans_temp(self):
        import minimal_loop
        registry, candidate, now, gate = self.activation_fixture()
        before = registry.current()
        log = (registry.root/"version_events.jsonl").read_bytes()
        actual_replace = minimal_loop.os.replace
        def fail_pointer(src, dst):
            if Path(dst) == registry.root/"current.json":
                raise OSError("controlled pointer replace failure")
            return actual_replace(src, dst)
        with patch("minimal_loop.os.replace", side_effect=fail_pointer):
            with self.assertRaisesRegex(OSError, "controlled pointer replace failure"):
                registry.publish(candidate, gate(), now, 0.90)
        self.assertEqual(registry.current(), before)
        self.assertEqual((registry.root/"version_events.jsonl").read_bytes(), log)
        self.assertFalse((registry.root/"current.json.tmp").exists())
        self.assertFalse((registry.root/"versions"/"M1").exists())

    def test_bootstrap_failure_leaves_no_pointer_and_can_retry(self):
        import minimal_loop
        registry, candidate, now, _ = self.activation_fixture(bootstrap=False)
        actual_replace = minimal_loop.os.replace
        def fail_pointer(src, dst):
            if Path(dst) == registry.root/"current.json":
                raise OSError("bootstrap pointer failure")
            return actual_replace(src, dst)
        with patch("minimal_loop.os.replace", side_effect=fail_pointer):
            with self.assertRaisesRegex(OSError, "bootstrap pointer failure"):
                registry.publish(candidate, None, now, 1.0)
        for name in ("current.json", "current.json.tmp", "version_events.jsonl", "versions/M0"):
            self.assertFalse((registry.root/name).exists(), name)
        self.assertEqual(registry.publish(candidate, None, now, 1.0)["model_version"], "M0")

    def test_simulator_failed_activation_keeps_runtime_and_disk_consistent(self):
        import minimal_loop
        actual_gate, actual_append = minimal_loop.model_gate_tool, minimal_loop.append_json
        def controlled_gate(*args, **kwargs):
            result = actual_gate(*args, **kwargs)
            result.update(model_gate(1.0, 0.90, 1.0, 1.0))
            return result
        def fail_upgrade_event(path, obj):
            if path.name == "version_events.jsonl" and obj["action"] == "upgrade":
                raise OSError("simulated activation log failure")
            return actual_append(path, obj)
        out = self.root/"failed_activation_simulation"
        with patch("minimal_loop.model_gate_tool", side_effect=controlled_gate), \
                patch("minimal_loop.append_json", side_effect=fail_upgrade_event):
            summary = simulate(self.data, self.times, out, max_episodes=31, exercise_at=30)
        self.assertEqual(summary["current"], VersionManager(out).current())
        self.assertEqual(summary["current"]["model_version"], "M0")
        self.assertEqual(summary["upgrades"], 0)
        self.assertEqual(len(summary["candidate_errors"]), 1)
        self.assertIn("simulated activation log failure", summary["candidate_errors"][0])
        predictions = [json.loads(s) for s in (out/"predictions.jsonl").read_text().splitlines()]
        reference = [json.loads(s) for s in (self.a/"predictions.jsonl").read_text().splitlines()]
        self.assertTrue(all(e["model_version"] == "M0" and e["activation_id"] == 0 for e in predictions))
        self.assertEqual([e["forecast"] for e in predictions], [e["forecast"] for e in reference[:len(predictions)]])
        self.assertEqual(len((out/"version_events.jsonl").read_text().splitlines()), 1)
        self.assertFalse((out/"versions"/"M1").exists())

    def test_simulator_successful_activation_uses_prepared_candidate(self):
        import minimal_loop
        actual_gate, actual_load = minimal_loop.model_gate_tool, VersionManager.load
        def controlled_gate(*args, **kwargs):
            result = actual_gate(*args, **kwargs)
            result.update(model_gate(1.0, 0.90, 1.0, 1.0))
            return result
        def reject_postcommit_load(manager, version):
            if (manager.root/"current.json").exists() and manager.current()["model_version"] == "M1":
                raise OSError("postcommit loading is forbidden in this test")
            return actual_load(manager, version)
        out = self.root/"successful_activation_simulation"
        with patch("minimal_loop.model_gate_tool", side_effect=controlled_gate), \
                patch.object(VersionManager, "load", new=reject_postcommit_load):
            summary = simulate(self.data, self.times, out, max_episodes=31, exercise_at=30)
        self.assertEqual(summary["candidate_errors"], [])
        self.assertEqual(summary["upgrades"], 1)
        self.assertEqual(summary["current"], VersionManager(out).current())
        ep = json.loads((out/"predictions.jsonl").read_text().splitlines()[-1])
        self.assertEqual((ep["model_version"], ep["activation_id"]), ("M1", 1))
        candidate, _ = VersionManager(out).load("M1")
        expected = candidate.predict_batch(np.array(ep["history"])[None])[0]
        np.testing.assert_allclose(ep["forecast"], expected, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
