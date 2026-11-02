from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .agent_gate import evaluate_agent_candidate
from .episode_store import EpisodeStore
from .knowledge_catalog import KnowledgeCatalog
from .model_gate import ModelGateConfig, evaluate_model_candidate
from .release_registry import JointReleaseRegistry
from .runtime_guard import RuntimeGuardConfig, runtime_safety_decision
from .common import write_json_atomic


def run_demo(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)

    # 1) Capture a diagnostic episode, then append a human-reviewed correction.
    episode_store = EpisodeStore(out / "experience" / "episodes.jsonl")
    episode = {
        "episode_id": "DEMO-E0001",
        "prediction_time": "2026-10-31T09:00:00+08:00",
        "equipment_id": "DEMO-PUMP-01",
        "operating_condition": "high_load",
        "fault_type": "bearing_fault_candidate",
        "task_type": "fault_amplitude_regression_and_diagnosis",
        "model": {"model_name": "LSTM", "model_version": "M0", "activation_id": 1,
                  "prediction": {"fault_amplitude": 0.42}, "confidence": 0.82,
                  "confidence_calibrated": False, "ood_score": 0.18,
                  "input_fingerprint": "demo-input-sha256"},
        "agent": {"release_id": "R0", "policy_version": "policy-0", "prompt_version": "prompt-0",
                  "tool_schema_version": "tools-0", "kb_version": "kb-demo-0",
                  "decision": "possible_bearing_fault", "rationale": "demo record only",
                  "citations": [{"source_id": "DOC-DEMO-01", "source_version": "1.0"}],
                  "tool_calls": [{"name": "retrieve_manual", "success": True,
                                  "input_ref": "query-demo", "output_ref": "result-demo"}]},
        "label": {"status": "pending", "value": None, "source_ref": None,
                  "reviewer": None, "reviewed_at": None},
        "failure_categories": [],
    }
    episode_store.append_episode(episode)
    episode_store.review_episode(
        "DEMO-E0001", status="verified", reviewer="demo-reviewer",
        source_ref="demo://work-order/WO-DEMO-01", ground_truth={"fault_type": "bearing_fault"},
        failure_categories=["agent_overconfidence", "evidence_or_citation_error"],
        corrected_decision="bearing_fault_confirmed",
        feedback_signals={"human_rating": -1, "diagnosis_overturned": True,
                          "user_feedback": "结论需要人工核实", "tool_failures": []},
        note="Synthetic example for schema verification; not a real maintenance record.",
    )

    # 2) Show why overall metrics are insufficient: a critical fault stratum worsens
    # while the global metrics improve on the same held-out samples.
    n = 240
    y_true = np.zeros(n, dtype=float)
    incumbent_errors = np.full(n, 0.5)
    candidate_errors = np.full(n, 0.05)
    fault_types = np.array(["normal"] * 220 + ["bearing_fault"] * 20)
    incumbent_errors[-20:] = 0.2
    candidate_errors[-20:] = 0.4  # locally worse on the critical fault type
    metadata = [{"equipment_id": "PUMP-A" if i % 2 == 0 else "PUMP-B",
                 "operating_condition": "high_load" if i % 3 == 0 else "normal_load",
                 "fault_type": str(fault_types[i])} for i in range(n)]
    model_gate = evaluate_model_candidate(
        y_true, incumbent_errors, candidate_errors, metadata,
        ModelGateConfig(critical_fault_types=("bearing_fault",)),
    )

    # 3) Replay a frozen Agent case. The evaluator checks explicit expectations;
    # it does not use an LLM as its own judge.
    replay_cases = [{"case_id": "CASE-CRITICAL-01", "allowed_decisions": ["manual_review"],
                     "required_tools": ["retrieve_manual"], "citations_required": True,
                     "manual_review_expected": True, "critical": True}]
    replay_outputs = [{"decision": "manual_review", "manual_review_requested": True,
                       "tool_calls": [{"name": "retrieve_manual", "success": True}],
                       "citations": [{"source_id": "DOC-01", "source_version": "3.2"}]}]
    agent_gate = evaluate_agent_candidate(replay_cases, replay_outputs)

    # 4) Runtime safety guard: no calibrated confidence => hard human review.
    runtime_guard = runtime_safety_decision(
        ood_score=0.18, drift_score=0.08, confidence=0.82,
        confidence_calibrated=False,
        config=RuntimeGuardConfig(calibration_id="DEMO-ONLY-NOT-FOR-PRODUCTION"),
    )

    # 5) Knowledge starts in draft and becomes retrievable only after review.
    knowledge = KnowledgeCatalog(out / "knowledge" / "catalog.jsonl")
    knowledge.register_document({
        "document_id": "DOC-DEMO-01", "version": "1.0", "title": "泵设备手册条目示例",
        "source_uri": "demo://manual/pump/section-1", "equipment_scope": ["DEMO-PUMP-01"],
        "effective_from": "2026-01-01", "effective_to": None,
        "summary": "演示用元数据，不包含可执行维修指令。正式知识需要厂家手册或审核 SOP。",
        "content": "泵设备 轴承 温升 振动 检查记录 演示条目。实际诊断应引用设备厂家手册和经审批维修规程。",
        "keywords": ["泵", "轴承", "振动", "温升"], "registered_by": "demo-author",
    })
    draft_hits = knowledge.search("泵 轴承 振动", equipment_id="DEMO-PUMP-01")
    knowledge.approve_document("DOC-DEMO-01", "1.0", reviewer="demo-reviewer",
                               approval_ref="demo://approval/APP-001")
    approved_hits = knowledge.search("泵 轴承 振动", equipment_id="DEMO-PUMP-01")

    # 6) Register joint release, but do not activate a rejected model candidate.
    registry = JointReleaseRegistry(out / "registry")
    checkpoint_sha = "a" * 64  # demo fixture only; replace with real checkpoint SHA-256
    manifest = registry.register_release({
        "release_id": "R-DEMO-001", "model_version": "M0", "model_activation_id": 1,
        "model_checkpoint_sha256": checkpoint_sha, "agent_policy_version": "policy-0",
        "prompt_version": "prompt-0", "tool_schema_version": "tools-0",
        "kb_version": "kb-demo-1", "retrieval_config_version": "tfidf-0",
        "risk_policy_version": "risk-0", "code_version": "demo-commit",
        "gate_evidence": {"model_gate_ref": "demo://gates/model",
                          "agent_gate_ref": "demo://gates/agent",
                          "shadow_ref": "demo://shadow/not-run"},
    })
    # Gate is intentionally not activated because this demonstration model gate rejects.

    report = {
        "demo_only": True,
        "episode": {"integrity": episode_store.verify_integrity(),
                    "reviewed": episode_store.get_episode("DEMO-E0001")["label"]},
        "model_gate": {"status": model_gate["status"], "allow_upgrade": model_gate["allow_upgrade"],
                       "global_mae_incumbent": model_gate["global"]["incumbent"]["mae"],
                       "global_mae_candidate": model_gate["global"]["candidate"]["mae"],
                       "bearing_fault_stratum": model_gate["strata"]["fault_type"]["bearing_fault"]["status"]},
        "agent_gate": {"status": agent_gate["status"], "metrics": agent_gate["metrics"]},
        "runtime_guard": runtime_guard,
        "knowledge": {"draft_search_hits": len(draft_hits), "approved_search_hits": len(approved_hits),
                      "first_hit": approved_hits[0] if approved_hits else None,
                      "integrity": knowledge.verify_integrity()},
        "joint_registry": {"registered_release": manifest["release_id"],
                           "status": manifest["status"], "current": registry.current(),
                           "integrity": registry.verify_integrity()},
        "interpretation": [
            "Global regression metrics improved, but the critical bearing_fault stratum worsened; candidate is rejected.",
            "Agent replay passed only against explicit frozen expectations; no prompt was self-modified.",
            "Uncalibrated confidence forces human review even when the numeric confidence is high.",
            "Draft knowledge is not retrievable; approval is required before RAG use.",
            "The joint release is registered as a candidate but not activated because model gate did not pass.",
        ],
    }
    write_json_atomic(out / "demo_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the V0 dual-evolution demo (synthetic fixtures only).")
    parser.add_argument("--out", default="dual_evolution_demo")
    args = parser.parse_args()
    report = run_demo(Path(args.out))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
