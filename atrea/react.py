"""Auditable ReAct event log (summaries only; never stores private chain-of-thought)."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


class ReActTrace:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, *, phase, action, observation=None, status="ok", artifacts=None, details=None):
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "phase": phase,
            "action": action,
            "observation": observation,
            "status": status,
            "artifacts": artifacts or [],
            "details": details or {},
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        payload["event_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        return payload
