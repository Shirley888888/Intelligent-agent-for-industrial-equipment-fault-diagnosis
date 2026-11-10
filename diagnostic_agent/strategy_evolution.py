"""Generate reviewable Agent strategy candidates from verified human corrections.

This module proposes versioned patches only. It never edits live prompts, tool
routing or retrieval configuration; candidates require frozen replay and approval.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dual_evolution.common import canonical_json, sha256_json


PROMPT_RULES = {
    "agent_decision_error": {
        "section": "decision_boundary",
        "instruction": "不得超出独立模型已提供的任务输出作出故障分类；模型未提供故障类别时必须明确说明并升级人工复核。",
        "replay_assertions": ["fault_class_is_model_authoritative", "missing_fault_label_escalates"],
    },
    "agent_overconfidence": {
        "section": "uncertainty_and_escalation",
        "instruction": "当置信度未校准、OOD/漂移信号缺失或超过门限时，必须采用不确定性表述并请求人工复核。",
        "replay_assertions": ["uncalibrated_confidence_escalates", "ood_critical_escalates"],
    },
    "evidence_or_citation_error": {
        "section": "evidence_policy",
        "instruction": "每条检查建议都必须引用本次检索实际返回且已审核的来源；无来源时不得生成具体检查步骤。",
        "replay_assertions": ["unretrieved_source_rejected", "empty_evidence_means_no_checks"],
    },
    "workflow_or_escalation_error": {
        "section": "workflow_policy",
        "instruction": "遇到关键证据缺失、工具失败或诊断结论被推翻时，停止给出确定性结论并创建人工复核任务。",
        "replay_assertions": ["critical_failure_escalates", "tool_error_escalates"],
    },
}

ROUTE_RULES = {
    "tool_routing_error": {
        "policy": "read_only_evidence_first",
        "ordered_tools": ["search_knowledge", "summarize_sensor_window"],
        "on_unknown_tool": "reject_and_escalate",
        "on_tool_error": "record_and_escalate",
    },
    "tool_execution_error": {
        "policy": "bounded_read_only_retry_then_escalate",
        "max_retry": 0,
        "retry_only_idempotent_read_tools": True,
        "on_tool_error": "record_and_escalate",
    },
}

RETRIEVAL_RULES = {
    "retrieval_miss": {
        "query_composition": ["equipment_id", "operating_condition", "model_fault_label", "human_verified_terms"],
        "top_k": 5,
        "filters": ["equipment_scope", "approved_only", "effective_on_prediction_date"],
        "fallback": "no_unverified_knowledge; escalate_if_diagnosis_requires_manual_source",
    },
    "knowledge_outdated_or_conflict": {
        "ranking_policy": "prefer_current_effective_approved_version",
        "conflict_policy": "surface_conflict_and_escalate",
        "filters": ["equipment_scope", "approved_only", "effective_on_prediction_date"],
    },
}


class StrategyCandidateGenerator:
    def __init__(self, path: str | Path, *, min_support: int = 2):
        if type(min_support) is not int or min_support < 2:
            raise ValueError("min_support must be an integer >= 2")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.min_support = min_support

    def _existing_ids(self) -> set[str]:
        ids = set()
        with self.path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    ids.add(row["candidate_id"])
        return ids

    @staticmethod
    def _eligible(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [x for x in episodes if x.get("label", {}).get("status") == "verified"]

    def generate(self, episodes: list[dict[str, Any]], *,
                 base_versions: dict[str, str] | None = None) -> list[dict[str, Any]]:
        verified = self._eligible(episodes)
        if not verified:
            return []
        base_versions = base_versions or {}
        existing = self._existing_ids()
        proposals: list[dict[str, Any]] = []
        category_map = {
            **{k: ("prompt", v) for k, v in PROMPT_RULES.items()},
            **{k: ("tool_route", v) for k, v in ROUTE_RULES.items()},
            **{k: ("retrieval", v) for k, v in RETRIEVAL_RULES.items()},
        }
        for category, (candidate_type, patch) in category_map.items():
            matches = [ep for ep in verified if category in ep.get("failure_categories", [])]
            if len(matches) < self.min_support:
                continue
            # Use the current policy version in the identity to avoid mixing unrelated baselines.
            policy_versions = sorted({str(ep.get("agent", {}).get("policy_version", "unknown")) for ep in matches})
            task_types = sorted({str(ep.get("task_type", "unknown")) for ep in matches})
            evidence_ids = sorted(str(ep["episode_id"]) for ep in matches)
            body = {"candidate_type": candidate_type, "failure_category": category,
                    "evidence_episode_ids": evidence_ids, "base_versions": base_versions,
                    "policy_versions": policy_versions, "task_types": task_types,
                    "patch": patch}
            candidate_id = "SC-" + hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()[:16]
            if candidate_id in existing:
                continue
            candidate = {
                "candidate_id": candidate_id,
                "candidate_type": candidate_type,
                "failure_category": category,
                "support_count": len(matches),
                "evidence_episode_ids": evidence_ids,
                "policy_versions": policy_versions,
                "task_types": task_types,
                "base_versions": base_versions,
                "patch": patch,
                "status": "proposed",
                "requires_human_approval": True,
                "requires_frozen_replay": True,
                "applied_to_live_agent": False,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "candidate_sha256": sha256_json(body),
            }
            proposals.append(candidate)
            existing.add(candidate_id)
        if proposals:
            with self.path.open("a", encoding="utf-8") as stream:
                for candidate in proposals:
                    stream.write(canonical_json(candidate) + "\n")
                stream.flush()
        return proposals

    def list_candidates(self) -> list[dict[str, Any]]:
        with self.path.open("r", encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
