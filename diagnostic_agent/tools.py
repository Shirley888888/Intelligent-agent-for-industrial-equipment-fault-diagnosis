from __future__ import annotations

import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dual_evolution.knowledge_catalog import KnowledgeCatalog


TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "search_knowledge",
     "description": "Search only approved, effective knowledge documents for the current equipment. Read-only.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string", "minLength": 1, "maxLength": 512},
         "top_k": {"type": "integer", "minimum": 1, "maximum": 5}},
         "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "summarize_sensor_window",
     "description": "Calculate descriptive statistics for supplied sensor readings. This does not classify faults.",
     "parameters": {"type": "object", "properties": {
         "metrics": {"type": "array", "items": {"type": "string"}, "maxItems": 20}},
         "required": [], "additionalProperties": False}}},
]


class ToolExecutor:
    """Allowlisted read-only tools plus a local human-review queue.

    No shell, network, arbitrary filesystem, database mutation or equipment-control
    tools are exposed to the planner.
    """

    ALLOWED = {"search_knowledge", "summarize_sensor_window"}

    def __init__(self, knowledge: KnowledgeCatalog, workspace: str | Path):
        self.knowledge = knowledge
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.review_queue = self.workspace / "human_review_queue.jsonl"
        self.audit_path = self.workspace / "tool_audit.jsonl"

    @staticmethod
    def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def execute(self, name: str, arguments: Any, context: dict[str, Any]) -> dict[str, Any]:
        started = datetime.now(timezone.utc).isoformat()
        try:
            if name not in self.ALLOWED:
                raise ValueError(f"Tool is not allowlisted: {name}")
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be an object")
            if name == "search_knowledge":
                result = self._search_knowledge(arguments, context)
            elif name == "summarize_sensor_window":
                result = self._summarize_sensors(arguments, context)
            else:  # defensive: ALLOWED and dispatch must remain synchronized
                raise ValueError(f"No executor for tool: {name}")
            record = {"name": name, "success": True, "started_at": started,
                      "finished_at": datetime.now(timezone.utc).isoformat(),
                      "arguments": arguments, "result": result}
        except Exception as exc:  # tool faults are isolated; the Agent fails conservatively
            result = {"error": type(exc).__name__, "message": str(exc)[:500]}
            record = {"name": name, "success": False, "started_at": started,
                      "finished_at": datetime.now(timezone.utc).isoformat(),
                      "arguments": arguments if isinstance(arguments, dict) else {}, "result": result}
        self._append_jsonl(self.audit_path, record)
        return record

    def _search_knowledge(self, arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        if set(arguments) - {"query", "top_k"}:
            raise ValueError("Unexpected search_knowledge argument")
        query = arguments.get("query")
        top_k = arguments.get("top_k", 5)
        if not isinstance(query, str) or not query.strip() or len(query) > 512:
            raise ValueError("query must be a nonempty string <= 512 characters")
        if type(top_k) is not int or not 1 <= top_k <= 5:
            raise ValueError("top_k must be an integer in [1,5]")
        # Equipment and date scope come from trusted runtime context, not the LLM.
        hits = self.knowledge.search(query, equipment_id=context["equipment_id"],
                                     as_of=context["prediction_time"][:10], top_k=top_k)
        return {"hits": hits, "count": len(hits), "scope_equipment": context["equipment_id"],
                "note": "Only approved and effective documents are returned."}

    @staticmethod
    def _summarize_sensors(arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        if set(arguments) - {"metrics"}:
            raise ValueError("Unexpected summarize_sensor_window argument")
        selected = arguments.get("metrics")
        readings = context.get("sensor_readings", {})
        if selected is not None:
            if not isinstance(selected, list) or len(selected) > 20 or not all(isinstance(x, str) for x in selected):
                raise ValueError("metrics must be a list of at most 20 strings")
            readings = {k: v for k, v in readings.items() if k in selected}
        summary = {}
        for name, value in readings.items():
            values = value if isinstance(value, list) else [value]
            numeric = []
            for item in values:
                if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                    continue
                numeric.append(float(item))
            if numeric:
                summary[name] = {"count": len(numeric), "min": min(numeric),
                                 "max": max(numeric), "mean": sum(numeric) / len(numeric),
                                 "last": numeric[-1]}
        return {"metrics": summary, "count": len(summary),
                "note": "Descriptive statistics only; no fault classification is performed."}

    def request_human_review(self, episode_id: str, reason: str, priority: str = "high") -> dict[str, Any]:
        if priority not in {"normal", "high", "urgent"}:
            priority = "high"
        ticket = {"ticket_id": str(uuid.uuid4()), "episode_id": episode_id,
                  "reason": str(reason)[:1000], "priority": priority,
                  "status": "open", "created_at": datetime.now(timezone.utc).isoformat()}
        self._append_jsonl(self.review_queue, ticket)
        return ticket
