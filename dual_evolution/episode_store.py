from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import canonical_json, require, sha256_json


LABEL_STATES = {"pending", "verified", "disputed", "invalid"}
FAILURE_CATEGORIES = {
    "model_prediction_error", "model_uncertainty_or_ood", "input_data_quality_error",
    "agent_decision_error", "agent_overconfidence", "tool_routing_error",
    "tool_execution_error", "retrieval_miss", "knowledge_outdated_or_conflict",
    "evidence_or_citation_error", "workflow_or_escalation_error",
    "unknown_root_cause", "other",
}
ALLOWED_TRANSITIONS = {
    "pending": {"verified", "disputed", "invalid"},
    "disputed": {"verified", "invalid"},
    "verified": set(),
    "invalid": set(),
}


class EpisodeStore:
    """Append-only, hash-chained JSONL store for diagnostic episodes.

    V0 supports a single writer. It intentionally has no multi-process lock or
    database transaction manager; production deployment must add one.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _read_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        prev_hash = "0" * 64
        seen_ids: set[str] = set()
        with self.path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Episode event JSON damaged at line {line_no}") from exc
                event_hash = event.pop("event_hash", None)
                if event.get("prev_hash") != prev_hash:
                    raise ValueError(f"Episode hash-chain prev_hash mismatch at line {line_no}")
                expected = sha256_json(event)
                if event_hash != expected:
                    raise ValueError(f"Episode hash-chain digest mismatch at line {line_no}")
                if event["event_id"] in seen_ids:
                    raise ValueError(f"Duplicate event_id at line {line_no}")
                seen_ids.add(event["event_id"])
                event["event_hash"] = event_hash
                events.append(event)
                prev_hash = event_hash
        return events

    def _append(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        events = self._read_events()  # verify the existing ledger before appending
        body = {
            "sequence": len(events) + 1,
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "created_at": self._now(),
            "prev_hash": events[-1]["event_hash"] if events else "0" * 64,
            "payload": payload,
        }
        event = {**body, "event_hash": sha256_json(body)}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return event

    @staticmethod
    def _validate_episode(episode: dict[str, Any]) -> dict[str, Any]:
        required = ["episode_id", "prediction_time", "equipment_id", "operating_condition",
                    "fault_type", "task_type", "model", "agent"]
        for field in required:
            require(field in episode, f"Episode missing required field: {field}")
        require(isinstance(episode["episode_id"], str) and episode["episode_id"].strip(),
                "episode_id must be a nonempty string")
        require(isinstance(episode["model"], dict), "model must be an object")
        require(isinstance(episode["agent"], dict), "agent must be an object")
        for field in ("model_version", "activation_id"):
            require(field in episode["model"], f"model.{field} is required")
        for field in ("policy_version", "prompt_version", "tool_schema_version", "kb_version"):
            require(field in episode["agent"], f"agent.{field} is required")
        require(isinstance(episode["agent"].get("tool_calls", []), list),
                "agent.tool_calls must be a list")
        require(isinstance(episode["agent"].get("citations", []), list),
                "agent.citations must be a list")
        clean = dict(episode)
        clean.setdefault("schema_version", "episode.v0.1")
        clean.setdefault("failure_categories", [])
        clean.setdefault("feedback_signals", {"human_rating": None, "user_feedback": None,
                                               "tool_failures": [], "diagnosis_overturned": False})
        require(isinstance(clean["feedback_signals"], dict), "feedback_signals must be an object")
        clean.setdefault("label", {"status": "pending", "value": None,
                                    "source_ref": None, "reviewer": None,
                                    "reviewed_at": None})
        require(clean["label"].get("status") == "pending",
                "New episodes must enter with label.status='pending'")
        return clean

    def append_episode(self, episode: dict[str, Any]) -> dict[str, Any]:
        clean = self._validate_episode(episode)
        current = self.get_episode(clean["episode_id"])
        require(current is None, f"Episode already exists: {clean['episode_id']}")
        event = self._append("episode_created", {"episode": clean})
        return {"episode_id": clean["episode_id"], "event_hash": event["event_hash"]}

    def review_episode(self, episode_id: str, *, status: str, reviewer: str,
                       source_ref: str, ground_truth: Any = None,
                       failure_categories: list[str] | None = None,
                       corrected_decision: str | None = None,
                       feedback_signals: dict[str, Any] | None = None,
                       note: str = "") -> dict[str, Any]:
        require(status in LABEL_STATES - {"pending"}, "Review status must be verified/disputed/invalid")
        require(bool(reviewer.strip()), "reviewer is required")
        require(bool(source_ref.strip()), "source_ref is required for an auditable review")
        current = self.get_episode(episode_id)
        require(current is not None, f"Unknown episode_id: {episode_id}")
        old_status = current["label"]["status"]
        require(status in ALLOWED_TRANSITIONS[old_status],
                f"Illegal label transition: {old_status} -> {status}")
        if status == "verified":
            require(ground_truth is not None, "verified labels require ground_truth")
        cats = sorted(set(failure_categories or []))
        unknown_cats = set(cats) - FAILURE_CATEGORIES
        require(not unknown_cats, f"Unknown failure categories: {sorted(unknown_cats)}")
        signals = feedback_signals or {}
        require(isinstance(signals, dict), "feedback_signals must be an object")
        event = self._append("episode_reviewed", {
            "episode_id": episode_id,
            "from_status": old_status,
            "status": status,
            "reviewer": reviewer,
            "source_ref": source_ref,
            "ground_truth": ground_truth,
            "failure_categories": cats,
            "corrected_decision": corrected_decision,
            "feedback_signals": signals,
            "note": note,
            "reviewed_at": self._now(),
        })
        return {"episode_id": episode_id, "status": status,
                "failure_categories": cats, "event_hash": event["event_hash"]}

    def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        result = None
        for event in self._read_events():
            payload = event["payload"]
            if event["event_type"] == "episode_created" and payload["episode"]["episode_id"] == episode_id:
                result = dict(payload["episode"])
            elif event["event_type"] == "episode_reviewed" and payload["episode_id"] == episode_id:
                if result is None:
                    raise ValueError(f"Review event appears before episode: {episode_id}")
                result["label"] = {
                    "status": payload["status"], "value": payload["ground_truth"],
                    "source_ref": payload["source_ref"], "reviewer": payload["reviewer"],
                    "reviewed_at": payload["reviewed_at"],
                }
                result["failure_categories"] = payload["failure_categories"]
                result["corrected_decision"] = payload["corrected_decision"]
                result["feedback_signals"].update(payload.get("feedback_signals", {}))
                result["review_note"] = payload["note"]
        return result

    def list_episodes(self, *, status: str | None = None) -> list[dict[str, Any]]:
        ids = []
        for event in self._read_events():
            if event["event_type"] == "episode_created":
                ids.append(event["payload"]["episode"]["episode_id"])
        items = [self.get_episode(episode_id) for episode_id in ids]
        return [item for item in items if item is not None and
                (status is None or item["label"]["status"] == status)]

    def verify_integrity(self) -> dict[str, Any]:
        events = self._read_events()
        return {"valid": True, "event_count": len(events),
                "last_hash": events[-1]["event_hash"] if events else "0" * 64,
                "writer_model": "single_writer"}
