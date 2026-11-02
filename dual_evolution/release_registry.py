from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import canonical_json, require, sha256_json, write_json_atomic


REQUIRED_COMPONENTS = ("model_version", "model_activation_id", "model_checkpoint_sha256",
                       "agent_policy_version", "prompt_version", "tool_schema_version",
                       "kb_version", "retrieval_config_version", "risk_policy_version",
                       "code_version")


class JointReleaseRegistry:
    """Small single-writer registry for reproducible model+Agent releases."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "release_events.jsonl"
        self.current_path = self.root / "current_release.json"
        self.events_path.touch(exist_ok=True)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _events(self) -> list[dict[str, Any]]:
        events = []
        prev_hash = "0" * 64
        with self.events_path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                event = json.loads(line)
                event_hash = event.pop("event_hash", None)
                if event.get("prev_hash") != prev_hash or event_hash != sha256_json(event):
                    raise ValueError(f"Release registry hash-chain error at line {line_no}")
                event["event_hash"] = event_hash
                events.append(event)
                prev_hash = event_hash
        return events

    def _append(self, event_type: str, payload: dict[str, Any]) -> None:
        events = self._events()
        body = {"sequence": len(events) + 1, "event_id": str(uuid.uuid4()),
                "event_type": event_type, "created_at": self._now(),
                "prev_hash": events[-1]["event_hash"] if events else "0" * 64,
                "payload": payload}
        event = {**body, "event_hash": sha256_json(body)}
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _releases(self) -> dict[str, dict[str, Any]]:
        releases: dict[str, dict[str, Any]] = {}
        for event in self._events():
            if event["event_type"] == "release_registered":
                manifest = event["payload"]["manifest"]
                releases[manifest["release_id"]] = dict(manifest)
        return releases

    def register_release(self, manifest: dict[str, Any]) -> dict[str, Any]:
        for field in REQUIRED_COMPONENTS:
            require(field in manifest and manifest[field] is not None and str(manifest[field]).strip() != "",
                    f"Release manifest missing {field}")
        require(type(manifest["model_activation_id"]) is int and manifest["model_activation_id"] >= 0,
                "model_activation_id must be a nonnegative integer")
        digest = str(manifest["model_checkpoint_sha256"])
        require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest.lower()),
                "model_checkpoint_sha256 must be a SHA-256 hex digest")
        release_id = str(manifest.get("release_id", "")).strip()
        require(bool(release_id), "release_id is required")
        releases = self._releases()
        require(release_id not in releases, f"release_id already exists: {release_id}")
        gate_evidence = manifest.get("gate_evidence", {})
        require(isinstance(gate_evidence, dict), "gate_evidence must be an object")
        clean = dict(manifest)
        clean.setdefault("created_at", self._now())
        clean["status"] = "candidate"
        clean["manifest_sha256"] = sha256_json({k: v for k, v in clean.items()
                                                 if k not in ("manifest_sha256", "status")})
        self._append("release_registered", {"manifest": clean})
        return clean

    def activate(self, release_id: str, *, approver: str,
                 model_gate_status: str, agent_gate_status: str,
                 shadow_status: str, safety_policy_approved: bool,
                 approval_ref: str, reason: str = "") -> dict[str, Any]:
        releases = self._releases()
        require(release_id in releases, f"Unknown release: {release_id}")
        require(all(x == "PASS" for x in (model_gate_status, agent_gate_status, shadow_status)),
                "Model gate, Agent gate and shadow status must all be PASS")
        evidence = releases[release_id].get("gate_evidence", {})
        for field in ("model_gate_ref", "agent_gate_ref", "shadow_ref"):
            require(bool(str(evidence.get(field, "")).strip()),
                    f"Release manifest gate_evidence.{field} is required before activation")
        require(safety_policy_approved is True, "Safety policy approval is required")
        require(bool(approver.strip()) and bool(approval_ref.strip()),
                "approver and approval_ref are required")
        event_count = len(self._events()) + 1
        pointer = {"release_id": release_id, "activation_id": event_count,
                   "action": "activate", "activated_at": self._now(),
                   "approver": approver, "approval_ref": approval_ref,
                   "gate_status": {"model": model_gate_status, "agent": agent_gate_status,
                                   "shadow": shadow_status},
                   "gate_evidence": evidence, "safety_policy_approved": True,
                   "reason": reason}
        # The event is durable before the current pointer is atomically replaced.
        self._append("release_activated", pointer)
        write_json_atomic(self.current_path, pointer)
        return pointer

    def rollback(self, release_id: str, *, approver: str,
                 approval_ref: str, reason: str) -> dict[str, Any]:
        releases = self._releases()
        require(release_id in releases, f"Unknown rollback target: {release_id}")
        previously_activated = {event["payload"].get("release_id") for event in self._events()
                                if event["event_type"] in ("release_activated", "release_rolled_back")}
        require(release_id in previously_activated,
                "Rollback target must be a previously activated release, not an untested candidate")
        require(bool(approver.strip()) and bool(approval_ref.strip()) and bool(reason.strip()),
                "Rollback requires approver, approval_ref and reason")
        event_count = len(self._events()) + 1
        pointer = {"release_id": release_id, "activation_id": event_count,
                   "action": "rollback", "activated_at": self._now(),
                   "approver": approver, "approval_ref": approval_ref,
                   "reason": reason}
        self._append("release_rolled_back", pointer)
        write_json_atomic(self.current_path, pointer)
        return pointer

    def current(self) -> dict[str, Any] | None:
        # The append-only event log is the source of truth. If the process was
        # interrupted after fsync(event) but before replacing the pointer,
        # rebuild the pointer from the last activation/rollback event.
        events = self._events()
        latest = next((event["payload"] for event in reversed(events)
                       if event["event_type"] in ("release_activated", "release_rolled_back")), None)
        if latest is None:
            return None
        pointer = None
        if self.current_path.exists():
            pointer = json.loads(self.current_path.read_text(encoding="utf-8"))
        if pointer is None or pointer.get("activation_id") != latest.get("activation_id"):
            write_json_atomic(self.current_path, latest)
            pointer = latest
        return pointer

    def get_release(self, release_id: str) -> dict[str, Any] | None:
        return self._releases().get(release_id)

    def verify_integrity(self) -> dict[str, Any]:
        events = self._events()
        return {"valid": True, "event_count": len(events),
                "release_count": len(self._releases()),
                "last_hash": events[-1]["event_hash"] if events else "0" * 64,
                "current": self.current()}
