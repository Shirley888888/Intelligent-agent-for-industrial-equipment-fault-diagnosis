from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


class Planner(Protocol):
    """Minimal provider contract used by the Agent tool loop."""

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Return {tool_calls:[{id,name,arguments}], content:str|None}."""


@dataclass
class OpenAICompatiblePlanner:
    """OpenAI-compatible Chat Completions adapter using only stdlib urllib.

    API keys are read from environment and are never stored in Episode records.
    The adapter is optional: tests and offline demonstrations do not call a network.
    """

    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 25.0
    max_retries: int = 1
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "OpenAICompatiblePlanner":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        model = os.getenv("OPENAI_MODEL", "").strip()
        if not api_key or not model:
            raise ValueError("Set OPENAI_API_KEY and OPENAI_MODEL to use the OpenAI-compatible provider")
        return cls(model=model, api_key=api_key,
                   base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                   timeout_seconds=float(os.getenv("AGENT_LLM_TIMEOUT", "25")),
                   max_retries=int(os.getenv("AGENT_LLM_RETRIES", "1")))

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.api_key or not self.model:
            raise ValueError("LLM api_key/model cannot be empty")
        endpoint = self.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        payload = {"model": self.model, "messages": messages,
                   "temperature": self.temperature, "tool_choice": "auto"}
        if tools:
            payload["tools"] = tools
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(endpoint, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
                message = data["choices"][0]["message"]
                calls = []
                for raw in message.get("tool_calls", []) or []:
                    function = raw.get("function", {})
                    raw_args = function.get("arguments", "{}")
                    arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    calls.append({"id": raw.get("id", ""), "name": function.get("name", ""),
                                  "arguments": arguments})
                return {"tool_calls": calls, "content": message.get("content")}
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                    json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(0.25 * (2 ** attempt), 1.0))
        raise RuntimeError(f"LLM provider request failed: {type(last_error).__name__}: {last_error}")


class OfflinePlanner:
    """Deterministic planner for local runs, CI and reproducible evaluation."""

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        user_payload = {}
        for message in messages:
            if message.get("role") == "user":
                try:
                    user_payload = json.loads(message.get("content", "{}"))
                except json.JSONDecodeError:
                    user_payload = {}
                break
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        called = {m.get("name") for m in tool_messages}
        if "search_knowledge" not in called:
            prediction = user_payload.get("model_prediction", {})
            label = prediction.get("fault_type")
            task_hint = prediction.get("target_name") or prediction.get("prediction_task") or "设备状态"
            equipment = user_payload.get("equipment_id", "unknown_equipment")
            condition = user_payload.get("operating_condition", "unknown_condition")
            query_hint = label if label else task_hint
            query = f"{equipment} {condition} {query_hint} 设备检查 维修手册"
            return {"tool_calls": [{"id": "offline-search-1", "name": "search_knowledge",
                                    "arguments": {"query": query, "top_k": 5}}], "content": None}
        if "summarize_sensor_window" not in called:
            return {"tool_calls": [{"id": "offline-sensors-1", "name": "summarize_sensor_window",
                                    "arguments": {"metrics": list((user_payload.get("sensor_readings") or {}).keys())[:20]}}],
                    "content": None}
        successful = []
        for message in tool_messages:
            if message.get("name") == "search_knowledge":
                try:
                    result = json.loads(message.get("content", "{}"))
                    successful = result.get("hits", [])
                except json.JSONDecodeError:
                    successful = []
        sources = [item.get("document_id") for item in successful if item.get("document_id")]
        content = {
            "observations": ["已读取独立模型输出和传感器统计摘要；Agent 未重新判定故障类别。"],
            "suggested_checks": ([{"check": "由具备资质的维护人员对照已审核资料检查相关状态，具体操作以现场安全规程为准",
                                   "source_ids": sources[:2]}] if sources else []),
            "limitations": ([] if sources else ["未检索到适用于该设备且已审核的维修知识；不生成维修步骤。"]),
        }
        return {"tool_calls": [], "content": json.dumps(content, ensure_ascii=False)}
