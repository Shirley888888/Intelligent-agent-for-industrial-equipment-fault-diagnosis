from __future__ import annotations

import json
import math
import uuid
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dual_evolution.episode_store import EpisodeStore
from dual_evolution.knowledge_catalog import KnowledgeCatalog
from dual_evolution.runtime_guard import RuntimeGuardConfig, runtime_safety_decision
from dual_evolution.common import sha256_json

from .prompts import PROMPT_VERSION, SYSTEM_PROMPT
from .providers import Planner
from .tools import TOOL_SCHEMAS, ToolExecutor
from .strategy_evolution import StrategyCandidateGenerator


@dataclass
class AgentRuntimeConfig:
    release_id: str = "local-unreleased"
    policy_version: str = "diagnostic-policy-v1.0.0"
    prompt_version: str = PROMPT_VERSION
    tool_schema_version: str = "diagnostic-tools-v1.0.0"
    kb_version: str = "knowledge-catalog-current"
    retrieval_config_version: str = "tfidf-v0"
    risk_policy_version: str = "runtime-guard-v0"
    max_tool_calls: int = 6
    max_llm_turns: int = 5
    strategy_min_support: int = 2
    runtime_guard: RuntimeGuardConfig | None = None

    def validate(self) -> None:
        if not 1 <= self.max_tool_calls <= 20:
            raise ValueError("max_tool_calls must be in [1,20]")
        if not 1 <= self.max_llm_turns <= 10:
            raise ValueError("max_llm_turns must be in [1,10]")
        if not self.release_id or not self.policy_version or not self.prompt_version:
            raise ValueError("Release/policy/prompt versions must be nonempty")
        if type(self.strategy_min_support) is not int or self.strategy_min_support < 2:
            raise ValueError("strategy_min_support must be an integer >= 2")


class DiagnosticAgent:
    """Tool-using diagnostic report Agent; fault class remains model-authoritative."""

    def __init__(self, *, planner: Planner, knowledge: KnowledgeCatalog,
                 episode_store: EpisodeStore, workspace: str | Path,
                 config: AgentRuntimeConfig | None = None):
        self.planner = planner
        self.knowledge = knowledge
        self.episode_store = episode_store
        self.workspace = Path(workspace)
        self.config = config or AgentRuntimeConfig()
        self.config.validate()
        self.tools = ToolExecutor(knowledge, self.workspace)
        self.strategy_candidates = StrategyCandidateGenerator(
            self.workspace / "strategy_candidates.jsonl",
            min_support=self.config.strategy_min_support)

    @staticmethod
    def _validate_context(raw: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError("Diagnostic context must be a JSON object")
        for field in ("equipment_id", "prediction_time", "operating_condition", "model", "model_prediction"):
            if field not in raw:
                raise ValueError(f"Missing required context field: {field}")
        model = raw["model"]
        prediction = raw["model_prediction"]
        if not isinstance(model, dict) or not isinstance(prediction, dict):
            raise ValueError("model and model_prediction must be objects")
        for field in ("model_version", "activation_id"):
            if field not in model:
                raise ValueError(f"model.{field} is required")
        if not isinstance(model["model_version"], str) or not model["model_version"].strip():
            raise ValueError("model.model_version must be nonempty")
        if type(model["activation_id"]) is not int or model["activation_id"] < 0:
            raise ValueError("model.activation_id must be a nonnegative integer")
        for field in ("equipment_id", "prediction_time", "operating_condition"):
            if not isinstance(raw[field], str) or not raw[field].strip():
                raise ValueError(f"{field} must be a nonempty string")
        try:
            datetime.fromisoformat(raw["prediction_time"].replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("prediction_time must be ISO-8601") from exc
        readings = raw.get("sensor_readings", {})
        if not isinstance(readings, dict) or len(readings) > 200:
            raise ValueError("sensor_readings must be an object with at most 200 metrics")
        for name, value in readings.items():
            if not isinstance(name, str) or len(name) > 100:
                raise ValueError("sensor metric names must be strings <= 100 chars")
            values = value if isinstance(value, list) else [value]
            if not isinstance(values, list) or len(values) > 10000:
                raise ValueError(f"sensor metric {name} must be a scalar or list of <=10000 values")
            for item in values:
                if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                    raise ValueError(f"sensor metric {name} contains a non-finite/non-numeric value")
        label = prediction.get("fault_type")
        if label is not None and (not isinstance(label, str) or len(label) > 128):
            raise ValueError("model_prediction.fault_type must be a string or null")
        if isinstance(label, str) and not label.strip():
            raise ValueError("model_prediction.fault_type cannot be an empty string")
        clean = dict(raw)
        clean["episode_id"] = str(raw.get("episode_id") or f"EP-{uuid.uuid4()}")
        clean["sensor_readings"] = readings
        clean["model"] = dict(model)
        clean["model_prediction"] = dict(prediction)
        clean.setdefault("task_type", "industrial_fault_diagnosis")
        clean.setdefault("input_quality", prediction.get("input_quality", "valid"))
        forecast_values = prediction.get("forecast_values")
        if forecast_values is not None:
            if not isinstance(forecast_values, list) or not 1 <= len(forecast_values) <= 512:
                raise ValueError("model_prediction.forecast_values must contain 1..512 values")
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x))
                   for x in forecast_values):
                raise ValueError("model_prediction.forecast_values must be finite numeric values")
        forecast_times = prediction.get("forecast_times")
        if forecast_times is not None and (not isinstance(forecast_times, list) or len(forecast_times) > 512
                or not all(isinstance(x, str) for x in forecast_times)):
            raise ValueError("model_prediction.forecast_times must be a list of at most 512 strings")
        # Chronos-2 oil forecasts include quantile paths; validate them before the LLM sees them.
        for quantile_name in ("forecast_p10", "forecast_p50", "forecast_p90"):
            quantile_values = prediction.get(quantile_name)
            if quantile_values is not None:
                if not isinstance(quantile_values, list) or len(quantile_values) > 512 or not quantile_values:
                    raise ValueError(f"model_prediction.{quantile_name} must be a non-empty list of <=512 values")
                if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x))
                       for x in quantile_values):
                    raise ValueError(f"model_prediction.{quantile_name} must contain finite numeric values")
        q_levels = prediction.get("forecast_quantile_levels")
        if q_levels is not None and (not isinstance(q_levels, list) or len(q_levels) > 20
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not 0 < float(x) < 1 for x in q_levels)):
            raise ValueError("model_prediction.forecast_quantile_levels must contain probabilities in (0,1)")
        clean.setdefault("sensor_conflict", False)
        return clean

    def _runtime_guard(self, context: dict[str, Any]) -> dict[str, Any]:
        p = context["model_prediction"]
        cfg = self.config.runtime_guard or RuntimeGuardConfig()
        calibration_flag = p.get("confidence_calibrated") is True
        # Industrial checkpoints carry a validation-derived calibration_id. It must
        # match the runtime policy configuration; a mismatch forces human review.
        model_calibration_id = p.get("calibration_id")
        if "calibration_id" in p:
            calibration_flag = bool(calibration_flag and model_calibration_id
                                     and model_calibration_id == cfg.calibration_id
                                     and cfg.calibration_id != "UNSET")
        return runtime_safety_decision(
            ood_score=p.get("ood_score"), drift_score=p.get("drift_score"),
            confidence=p.get("confidence"), confidence_calibrated=calibration_flag,
            input_quality=context.get("input_quality", "valid"),
            sensor_conflict=context.get("sensor_conflict") is True, config=cfg)

    @staticmethod
    def _tool_result_content(record: dict[str, Any]) -> str:
        return json.dumps(record.get("result", {}), ensure_ascii=False, allow_nan=False)

    @staticmethod
    def _parse_final(content: Any) -> dict[str, Any]:
        if not isinstance(content, str) or not content.strip():
            return {"observations": [], "suggested_checks": [],
                    "limitations": ["Agent 未返回结构化报告内容。"]}
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {"observations": [], "suggested_checks": [],
                    "limitations": ["Agent 输出不是有效 JSON，已丢弃自由文本并采用安全模板。"]}
        if not isinstance(parsed, dict):
            return {"observations": [], "suggested_checks": [],
                    "limitations": ["Agent 输出结构不合法，已采用安全模板。"]}
        def strings(name: str, limit: int) -> list[str]:
            values = parsed.get(name, [])
            if not isinstance(values, list):
                return []
            return [str(x)[:limit] for x in values[:10] if isinstance(x, str) and x.strip()]
        checks = []
        raw_checks = parsed.get("suggested_checks", [])
        if isinstance(raw_checks, list):
            for item in raw_checks[:5]:
                if isinstance(item, dict) and isinstance(item.get("check"), str) and isinstance(item.get("source_ids"), list):
                    checks.append({"check": item["check"][:500],
                                   "source_ids": [str(x)[:128] for x in item["source_ids"][:5] if isinstance(x, str)]})
        return {"observations": strings("observations", 500),
                "suggested_checks": checks, "limitations": strings("limitations", 500)}

    def run(self, raw_context: dict[str, Any]) -> dict[str, Any]:
        context = self._validate_context(raw_context)
        guard = self._runtime_guard(context)
        prediction = context["model_prediction"]
        model_fault_type = prediction.get("fault_type")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({
                "equipment_id": context["equipment_id"],
                "prediction_time": context["prediction_time"],
                "operating_condition": context["operating_condition"],
                "sensor_readings": context["sensor_readings"],
                "model": {k: context["model"].get(k) for k in ("model_name", "model_version", "activation_id")},
                "task_type": context["task_type"],
                "model_prediction": {k: prediction.get(k) for k in (
                    "fault_type", "fault_class_index", "fault_probabilities", "anomaly_score",
                    "confidence", "confidence_calibrated", "calibration_id", "ood_score",
                    "drift_score", "rul_hours", "rul_unit", "fault_amplitude",
                    "fault_amplitude_unit", "input_quality", "prediction_task", "prediction_kind",
                    "target_name", "target_unit", "forecast_horizon", "forecast_values",
                    "forecast_times", "forecast_summary", "forecast_quantile_levels",
                    "forecast_quantiles", "forecast_p10", "forecast_p50", "forecast_p90") if k in prediction},
                "runtime_safety": guard,
            }, ensure_ascii=False, allow_nan=False)},
        ]
        tool_calls: list[dict[str, Any]] = []
        search_hits: list[dict[str, Any]] = []
        provider_error: str | None = None
        final_content: Any = None
        for turn in range(self.config.max_llm_turns):
            try:
                response = self.planner.complete(messages, TOOL_SCHEMAS)
                if not isinstance(response, dict):
                    raise TypeError("Planner response must be an object")
                calls = response.get("tool_calls", []) or []
                if not isinstance(calls, list):
                    raise TypeError("tool_calls must be a list")
                if calls:
                    assistant_calls = []
                    for idx, call in enumerate(calls):
                        if not isinstance(call, dict):
                            tool_calls.append({"name": "malformed_tool_call", "success": False,
                                               "arguments": {}, "result": {"error": "tool_call_must_be_object"}})
                            continue
                        name = str(call.get("name", ""))
                        args = call.get("arguments", {})
                        call_id = str(call.get("id") or f"tool-{turn}-{idx}")
                        if len(tool_calls) >= self.config.max_tool_calls:
                            tool_calls.append({"name": name or "unknown", "success": False,
                                               "arguments": {}, "result": {"error": "tool_call_budget_exceeded"}})
                            break
                        record = self.tools.execute(name, args, context)
                        record["call_id"] = call_id
                        tool_calls.append(record)
                        assistant_calls.append({"id": call_id, "type": "function", "function": {
                            "name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
                        if name == "search_knowledge" and record.get("success"):
                            search_hits = record["result"].get("hits", [])
                    messages.append({"role": "assistant", "content": None, "tool_calls": assistant_calls})
                    for record in tool_calls[-len(assistant_calls):] if assistant_calls else []:
                        messages.append({"role": "tool", "tool_call_id": record.get("call_id", ""),
                                         "name": record.get("name", "unknown"),
                                         "content": self._tool_result_content(record)})
                    if any(not item.get("success") for item in tool_calls):
                        # Tool errors are preserved in the trace. Agent may continue, but release is conservative.
                        pass
                    continue
                final_content = response.get("content")
                break
            except Exception as exc:
                provider_error = f"{type(exc).__name__}: {str(exc)[:400]}"
                break
        if final_content is None and provider_error is None:
            provider_error = "Planner did not return a final report within max_llm_turns"

        parsed = self._parse_final(final_content)
        valid_source_ids = {str(hit.get("document_id")) for hit in search_hits}
        safe_checks = []
        for check in parsed["suggested_checks"]:
            source_ids = sorted(set(check["source_ids"]) & valid_source_ids)
            if source_ids and check["check"].strip():
                safe_checks.append({"check": check["check"], "source_ids": source_ids,
                                    "status": "professional_review_required"})
        tool_failures = [x for x in tool_calls if not x.get("success")]
        failure_categories: list[str] = []
        if any(reason in guard.get("reasons", []) for reason in
               ("critical_ood", "elevated_ood", "critical_distribution_drift", "distribution_drift",
                "low_confidence", "confidence_not_calibrated", "calibration_not_configured")):
            failure_categories.append("model_uncertainty_or_ood")
        if tool_failures:
            failure_categories.append("tool_execution_error")
        if not search_hits:
            failure_categories.append("retrieval_miss")
        if provider_error:
            failure_categories.append("agent_decision_error")
        if parsed["suggested_checks"] and not safe_checks:
            failure_categories.append("evidence_or_citation_error")
        reasons = list(guard.get("reasons", []))
        if model_fault_type is None:
            reasons.append("independent_model_fault_label_missing")
        if provider_error:
            reasons.append("planner_or_tool_loop_error")
        if tool_failures:
            reasons.append("tool_execution_failure")
        request_review = bool(guard.get("request_human_review")) or model_fault_type is None or bool(provider_error) or bool(tool_failures)
        if request_review:
            decision = "human_review"
        elif not search_hits:
            decision = "model_prediction_only"
        else:
            decision = "provisional_model_result"
        episode_id = context["episode_id"]
        ticket = None
        if request_review:
            ticket = self.tools.request_human_review(
                episode_id, ";".join(reasons) or "manual_review_requested",
                "urgent" if "critical_ood" in reasons or "critical_distribution_drift" in reasons else "high")

        # The authoritative fault class is copied from the independent model only.
        report = {
            "episode_id": episode_id,
            "equipment_id": context["equipment_id"],
            "prediction_time": context["prediction_time"],
            "decision": decision,
            "manual_review_requested": request_review,
            "diagnosis": {
                "fault_type": model_fault_type,
                "fault_type_source": ("independent_industrial_fault_model"
                                      if prediction.get("prediction_task") == "industrial_fault_diagnosis"
                                      else "independent_time_series_model"),
                "fault_type_overridable_by_agent": False,
                "model_version": context["model"]["model_version"],
                "activation_id": context["model"]["activation_id"],
                "prediction": {k: prediction.get(k) for k in
                               ("fault_class_index", "fault_probabilities", "anomaly_score", "confidence",
                                "confidence_calibrated", "calibration_id", "ood_score", "drift_score",
                                "rul_hours", "rul_unit", "fault_amplitude", "fault_amplitude_unit",
                                "prediction_task", "prediction_kind", "target_name", "target_unit",
                                "forecast_horizon", "forecast_values", "forecast_times", "forecast_summary",
                                "forecast_quantile_levels", "forecast_quantiles", "forecast_p10",
                                "forecast_p50", "forecast_p90")
                               if k in prediction},
            },
            "summary": (f"独立工业故障模型输出类别：{model_fault_type}。Agent 未重新分类；"
                        f"当前处理状态：{decision}。" if model_fault_type and context.get("task_type") == "industrial_fault_diagnosis" else
                        (f"独立时序模型输出候选故障类别：{model_fault_type}。Agent 未重新分类；"
                         f"当前处理状态：{decision}。" if model_fault_type else
                        (f"独立模型完成 {prediction.get('prediction_task', '时序预测')}，"
                         f"目标变量为 {prediction.get('target_name', '未指定')}；当前模型未输出故障类别，"
                         "Agent 不从回归预测推断故障类别，已请求人工复核。"
                         if prediction.get("prediction_kind") == "regression" else
                         "独立模型未提供故障类别，Agent 不进行补充猜测，已请求人工复核。"))),
            "observations": parsed["observations"],
            "suggested_checks": safe_checks,
            "limitations": parsed["limitations"] + (["维修检查建议仅来自当前命中的已审核知识条目，仍须由专业人员按现场规程核验。"] if safe_checks else []),
            "citations": [{"source_id": h["document_id"], "source_version": h["version"],
                           "title": h["title"], "source_uri": h["source_uri"],
                           "content_sha256": h["content_sha256"], "score": h["score"],
                           "snippet": h["snippet"]} for h in search_hits],
            "runtime_safety": guard,
            "failure_categories": failure_categories,
            "tool_calls": [{k: v for k, v in item.items() if k not in ("result", "arguments")} |
                           {"arguments": item.get("arguments", {}), "result": item.get("result", {})}
                           for item in tool_calls],
            "human_review_ticket": ticket,
            "provider_error": provider_error,
            "versions": {"release_id": self.config.release_id,
                         "policy_version": self.config.policy_version,
                         "prompt_version": self.config.prompt_version,
                         "tool_schema_version": self.config.tool_schema_version,
                         "kb_version": self.config.kb_version,
                         "retrieval_config_version": self.config.retrieval_config_version,
                         "risk_policy_version": self.config.risk_policy_version,
                         "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                         "tool_schema_sha256": sha256_json(TOOL_SCHEMAS)},
        }
        episode = {
            "episode_id": episode_id,
            "prediction_time": context["prediction_time"],
            "equipment_id": context["equipment_id"],
            "operating_condition": context["operating_condition"],
            "fault_type": model_fault_type or "unknown_model_output",
            "task_type": context["task_type"],
            "model": {**context["model"], "prediction": prediction,
                      "confidence": prediction.get("confidence"),
                      "confidence_calibrated": prediction.get("confidence_calibrated"),
                      "ood_score": prediction.get("ood_score"),
                      "drift_score": prediction.get("drift_score")},
            "agent": {"release_id": self.config.release_id,
                      "policy_version": self.config.policy_version,
                      "prompt_version": self.config.prompt_version,
                      "tool_schema_version": self.config.tool_schema_version,
                      "kb_version": self.config.kb_version,
                      "retrieval_config_version": self.config.retrieval_config_version,
                      "risk_policy_version": self.config.risk_policy_version,
                      "prompt_sha256": report["versions"]["prompt_sha256"],
                      "tool_schema_sha256": report["versions"]["tool_schema_sha256"],
                      "decision": decision, "rationale": report["summary"],
                      "citations": report["citations"], "tool_calls": report["tool_calls"],
                      "manual_review_requested": request_review},
            "failure_categories": failure_categories,
            "source_data": context.get("source_data"),
            "feedback_signals": {"human_rating": None, "user_feedback": None,
                                 "tool_failures": [x["name"] for x in tool_failures],
                                 "diagnosis_overturned": False},
        }
        episode_write = self.episode_store.append_episode(episode)
        report["episode_persistence"] = {"stored": True, **episode_write}
        return report

    def submit_feedback(self, episode_id: str, *, reviewer: str, source_ref: str,
                        status: str, ground_truth: Any = None,
                        corrected_decision: str | None = None,
                        failure_categories: list[str] | None = None,
                        note: str = "") -> dict[str, Any]:
        """Submit a human/authoritative outcome; pending labels never train directly."""
        review = self.episode_store.review_episode(
            episode_id, status=status, reviewer=reviewer, source_ref=source_ref,
            ground_truth=ground_truth, corrected_decision=corrected_decision,
            failure_categories=failure_categories, note=note,
            feedback_signals={"diagnosis_overturned": bool(corrected_decision),
                              "human_rating": -1 if corrected_decision else 1})
        # Only verified human corrections can generate candidates; proposals never
        # modify the live prompt, route or retrieval configuration.
        proposals = []
        if status == "verified":
            proposals = self.strategy_candidates.generate(
                self.episode_store.list_episodes(status="verified"),
                base_versions={"prompt_version": self.config.prompt_version,
                               "tool_schema_version": self.config.tool_schema_version,
                               "retrieval_config_version": self.config.retrieval_config_version,
                               "policy_version": self.config.policy_version})
        return {**review, "strategy_candidates_generated": len(proposals),
                "strategy_candidates": proposals,
                "strategy_candidate_store": str(self.strategy_candidates.path)}
