"""工业设备故障诊断智能体原型：模型诊断、知识检索、可验证证据链和受约束报告生成。"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any


CONFIDENCE_THRESHOLD = 0.70
DEFAULT_LLM_MODEL = "gpt-5.6-terra"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

# 模拟维护知识库。真实向量数据库中 source_id 应为不可变、可审计的文档标识。
KNOWLEDGE_BASE: list[dict[str, str]] = [
    {"fault_label": "bearing_wear", "source_id": "KB-BRG-001", "priority": "high",
     "content": "轴承磨损通常伴随振动升高和轴承温度上升；先检查润滑、游隙和紧固状态。"},
    {"fault_label": "bearing_wear", "source_id": "KB-BRG-002", "priority": "high",
     "content": "振动持续超限时应在计划停机窗口拆检；发现剥落、点蚀或保持架损坏时更换轴承并试运行。"},
    {"fault_label": "bearing_wear", "source_id": "KB-BRG-003", "priority": "medium",
     "content": "维护后应复测振动和温度趋势，确认指标恢复到设备基线范围。"},
    {"fault_label": "rotor_imbalance", "source_id": "KB-IMB-001", "priority": "high",
     "content": "转子不平衡常表现为与转速相关的周期性振动；检查积灰、附着物和配重状态。"},
    {"fault_label": "rotor_imbalance", "source_id": "KB-IMB-002", "priority": "high",
     "content": "停机后清理附着物，并依照设备规范完成动平衡校正。"},
    {"fault_label": "overheating", "source_id": "KB-OVH-001", "priority": "high",
     "content": "设备过热时优先核查冷却回路、润滑状态、负载工况与环境温度。"},
    {"fault_label": "overheating", "source_id": "KB-OVH-002", "priority": "critical",
     "content": "温度接近安全上限应降低负载；温度持续上升时应按安全规程停机。"},
    {"fault_label": "normal", "source_id": "KB-NRM-001", "priority": "low",
     "content": "未发现明确故障特征时，继续开展趋势监测、日常巡检和预防性维护。"},
]


def _series(sensor_data: dict[str, Any], name: str) -> list[float]:
    """输入：原始传感器字典与字段名；输出：校验后的浮点时序。"""
    values = sensor_data.get(name)
    if not isinstance(values, list) or not values:
        raise ValueError(f"sensor_data.{name} 必须是非空数值列表。")
    if any(not isinstance(value, (int, float)) for value in values):
        raise ValueError(f"sensor_data.{name} 含有非数值项。")
    return [float(value) for value in values]


def run_fault_prediction(sensor_data: dict[str, Any]) -> dict[str, Any]:
    """模拟独立时序故障预测模型。

    输入：带 device_id、vibration_rms、temperature、rotational_speed 的传感器时序数据。
    输出：结构化诊断 JSON，包含设备、故障标签、置信度、候选故障和异常指标。

    安全边界：这是系统内唯一产生 fault_label 的函数；LLM 不会调用或参与本函数。
    """
    device_id = sensor_data.get("device_id")
    if not isinstance(device_id, str) or not device_id.strip():
        raise ValueError("sensor_data.device_id 必须是非空字符串。")
    vibration, temperature, speed = (_series(sensor_data, key) for key in
                                              ("vibration_rms", "temperature", "rotational_speed"))
    vibration_max, temperature_max = max(vibration), max(temperature)
    sensor_scores = {
        "vibration_rms": round(min(vibration_max / 10, 1), 3),
        "temperature": round(min(max(temperature_max - 40, 0) / 60, 1), 3),
    }

    # 占位推理逻辑：后续应整体替换为真实模型/API，不能由 LLM 替代。
    if vibration_max >= 7 and temperature_max >= 80:
        label, confidence, abnormal = "bearing_wear", 0.91, ["vibration_rms", "temperature"]
        candidates = [("bearing_wear", 0.91), ("overheating", 0.72), ("rotor_imbalance", 0.55)]
    elif vibration_max >= 6:
        label, confidence, abnormal = "rotor_imbalance", 0.82, ["vibration_rms"]
        candidates = [("rotor_imbalance", 0.82), ("bearing_wear", 0.63)]
    elif temperature_max >= 85:
        label, confidence, abnormal = "overheating", 0.80, ["temperature"]
        candidates = [("overheating", 0.80), ("bearing_wear", 0.51)]
    elif vibration_max >= 4.5 or temperature_max >= 70:
        label, confidence, abnormal = "bearing_wear", 0.62, ["vibration_rms"]
        candidates = [("bearing_wear", 0.62), ("rotor_imbalance", 0.57)]
    else:
        label, confidence, abnormal = "normal", 0.95, []
        candidates = [("normal", 0.95), ("bearing_wear", 0.04)]

    return {
        "device_id": device_id,
        "fault_label": label,
        "fault_confidence": confidence,
        "candidate_fault": [{"fault_label": item[0], "confidence": item[1]} for item in candidates],
        "abnormal_sensor": abnormal,
        "sensor_abnormal_score": {key: sensor_scores[key] for key in abnormal},
        "model_feature_summary": {
            "vibration_rms_max": round(vibration_max, 3),
            "temperature_max": round(temperature_max, 3),
            "rotational_speed_mean": round(sum(speed) / len(speed), 3),
        },
        "model_name": "mock_temporal_fault_model_v1",
        "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def retrieve_maintain_knowledge(fault_label: str, top_k: int = 3) -> list[dict[str, str]]:
    """模拟向量知识库检索。

    输入：只能来自预测模型的 fault_label，以及返回数量 top_k。
    输出：含 content、source_id、fault_label、priority 的候选知识片段。
    """
    if not isinstance(fault_label, str) or not fault_label or top_k < 1:
        raise ValueError("fault_label 应为非空字符串，top_k 应为正整数。")
    matches = [item.copy() for item in KNOWLEDGE_BASE if item["fault_label"] == fault_label]
    # 仅为演示证据链过滤能力而加入一条无关候选记录。
    if matches and fault_label != "normal":
        matches.append({"fault_label": "unrelated", "source_id": "KB-TEST-999", "priority": "low",
                        "content": "此条记录故意用于验证无关检索结果过滤。"})
    return matches[:top_k]


def build_evidence_chain(model_result: dict[str, Any], knowledge_list: list[dict[str, str]],
                         conf_threshold: float = CONFIDENCE_THRESHOLD) -> dict[str, Any]:
    """构建可验证诊断证据链。

    输入：模型结构化结果、候选知识列表、置信度阈值。
    输出：含模型证据、经标签过滤后的知识证据、风险标记和可引用来源 ID 的字典。
    """
    required = {"device_id", "fault_label", "fault_confidence", "candidate_fault",
                "abnormal_sensor", "sensor_abnormal_score"}
    missing = required - model_result.keys()
    if missing:
        raise ValueError(f"模型输出缺少字段：{sorted(missing)}")
    confidence, label = float(model_result["fault_confidence"]), model_result["fault_label"]
    if not 0 <= confidence <= 1:
        raise ValueError("fault_confidence 必须位于 0 到 1。")
    relevant = [item for item in knowledge_list if item.get("fault_label") == label
                and item.get("source_id") and item.get("content")]
    filtered = [item.get("source_id", "UNKNOWN") for item in knowledge_list if item not in relevant]
    low_confidence, no_knowledge = confidence < conf_threshold, not relevant
    risks: list[dict[str, Any]] = []
    if low_confidence:
        risks.append({"risk_type": "LOW_MODEL_CONFIDENCE", "level": "high",
                      "message": f"模型置信度 {confidence:.2f} 低于阈值 {conf_threshold:.2f}，必须人工复核。",
                      "manual_review_required": True})
    if no_knowledge:
        risks.append({"risk_type": "MISSING_KNOWLEDGE_EVIDENCE", "level": "high",
                      "message": "缺少与模型标签匹配的维护知识，必须人工复核。",
                      "manual_review_required": True})
    if not risks:
        risks.append({"risk_type": "NO_BLOCKING_RISK", "level": "info",
                      "message": "模型置信度达标，且存在匹配的知识库证据。", "manual_review_required": False})
    return {
        "evidence_chain_id": f"EC-{model_result['device_id']}-{datetime.now(timezone.utc):%Y%m%d%H%M%S}",
        "model_evidence": {key: model_result.get(key) for key in (
            "device_id", "model_name", "fault_label", "fault_confidence", "candidate_fault",
            "abnormal_sensor", "sensor_abnormal_score", "model_feature_summary", "prediction_timestamp")},
        "knowledge_base_evidence": [{"source_id": item["source_id"], "content": item["content"],
                                       "priority": item["priority"]} for item in relevant],
        "risk_flags": risks,
        "verification": {"confidence_threshold": conf_threshold, "confidence_passed": not low_confidence,
                         "manual_review_required": any(r["manual_review_required"] for r in risks),
                         "allowed_source_ids": [item["source_id"] for item in relevant],
                         "filtered_out_source_ids": filtered},
    }


def build_report_prompt(model_result: dict[str, Any], evidence_chain: dict[str, Any],
                        knowledge_list: list[dict[str, str]]) -> str:
    """组装受限 Prompt；输入中刻意没有 sensor_data，确保 LLM 不见原始时序。"""
    permitted = set(evidence_chain["verification"]["allowed_source_ids"])
    verified_knowledge = [item for item in knowledge_list if item.get("source_id") in permitted]
    return f'''你是工业设备维护报告整理助手。

不可违反的规则：
1. 你不得自行推断、猜测、修改或重新判断故障；故障全部来自故障预测模型输出的 fault_label。
2. 维修建议必须引用 source_id，且仅可引用 allowed_source_ids。
3. 当 manual_review_required 为 true，必须在报告开头输出“人工复核警告”。
4. 仅输出 Markdown 处置报告，必须含模型诊断结果、故障解释、分级处置建议、风险提示、证据摘要。
5. 原始 sensor_data 未提供，禁止假造或索取其内容。

# MODEL_RESULT
{json.dumps(model_result, ensure_ascii=False)}
# EVIDENCE_CHAIN
{json.dumps(evidence_chain, ensure_ascii=False)}
# VERIFIED_KNOWLEDGE
{json.dumps(verified_knowledge, ensure_ascii=False)}'''


def _mock_call_llm(prompt: str) -> str:
    """模拟 LLM 接口：只整理 Prompt 中已验证材料，绝不进行故障判别。"""
    if "你不得自行推断、猜测、修改或重新判断故障" not in prompt:
        raise ValueError("Prompt 缺失 LLM 故障判别禁止约束。")
    def block(start: str, end: str | None = None) -> Any:
        value = prompt.split(start, 1)[1]
        if end:
            value = value.split(end, 1)[0]
        return json.loads(value.strip())
    model = block("# MODEL_RESULT", "# EVIDENCE_CHAIN")
    chain = block("# EVIDENCE_CHAIN", "# VERIFIED_KNOWLEDGE")
    knowledge = block("# VERIFIED_KNOWLEDGE")
    refs = [item["source_id"] for item in knowledge]
    ref = refs[0] if refs else "无有效来源"
    sensor = "、".join(f"{name}（评分 {model['sensor_abnormal_score'][name]:.3f}）"
                         for name in model["abnormal_sensor"]) or "未发现显著异常传感器"
    report = ["# 工业设备故障诊断与处置报告", ""]
    if chain["verification"]["manual_review_required"]:
        report += ["## ⚠️ 人工复核警告（必须执行）", "",
                   "模型可信度或知识证据不足。禁止基于本报告自动执行高风险操作，必须由维护人员现场复核。", ""]
    report += ["## 模型诊断结果", "",
               f"- 设备编号：`{model['device_id']}`",
               f"- 故障标签（唯一模型来源）：`{model['fault_label']}`",
               f"- 模型置信度：`{model['fault_confidence']:.2f}`",
               f"- 异常传感器指标：{sensor}", "",
               "## 故障解释", "",
               f"本报告不自主判断故障类别，仅解释模型输出的 `{model['fault_label']}` 及其异常传感器证据。", "",
               "## 分级处置建议", "",
               "### 一级：立即核查", "",
               f"依据维护知识，先完成现场状态与相关部件检查。 [来源: {ref}]", "",
               "### 二级：计划维护", "",
               f"依据设备维护窗口安排进一步检修，并在处置后复测趋势指标。 [来源: {refs[-1] if refs else ref}]", "",
               "## 风险提示", ""]
    report += [f"- `{risk['level']}` / `{risk['risk_type']}`：{risk['message']}" for risk in chain["risk_flags"]]
    report += ["", "## 证据摘要", "",
               f"- 证据链编号：`{chain['evidence_chain_id']}`",
               f"- 模型置信度：`{model['fault_confidence']:.2f}`",
               f"- 异常传感器：{sensor}",
               f"- 知识库来源 ID：`{', '.join(refs) if refs else '无'}`",
               "- 诊断边界：故障类别只来自独立故障预测模型；LLM 仅生成可解释处置报告。"]
    return "\n".join(report)


def _validate_llm_report(report: str, evidence_chain: dict[str, Any]) -> None:
    """校验真实或模拟 LLM 报告是否保留硬性溯源和人工复核要求。"""
    required_sections = ("模型诊断结果", "故障解释", "分级处置建议", "风险提示", "证据摘要")
    missing_sections = [section for section in required_sections if section not in report]
    if missing_sections:
        raise ValueError(f"LLM 报告缺少必要章节：{missing_sections}")
    if evidence_chain["verification"]["manual_review_required"] and "人工复核警告" not in report:
        raise ValueError("低置信度/证据不足场景下，LLM 报告缺少强制人工复核警告。")
    advice_section = report.split("## 分级处置建议", 1)[1].split("## 风险提示", 1)[0]
    advice_lines = [line.strip() for line in advice_section.splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
    uncited_advice = [line for line in advice_lines if "[来源: " not in line]
    if uncited_advice:
        raise ValueError(f"LLM 报告存在未引用 source_id 的维修建议：{uncited_advice}")
    allowed_source_ids = evidence_chain["verification"]["allowed_source_ids"]
    if allowed_source_ids and not any(f"[来源: {source_id}]" in report for source_id in allowed_source_ids):
        raise ValueError("LLM 报告未引用任何已验证知识库 source_id。")
    forbidden_sources = set()
    for token in report.split("[来源: ")[1:]:
        source_id = token.split("]", 1)[0].strip()
        if source_id and source_id not in allowed_source_ids:
            forbidden_sources.add(source_id)
    if forbidden_sources:
        raise ValueError(f"LLM 报告引用了未授权 source_id：{sorted(forbidden_sources)}")


def _call_openai_responses(prompt: str) -> str:
    """调用 OpenAI Responses API；仅将受限 Prompt 发送给 LLM。"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 OPENAI_API_KEY。请设置 API 密钥后，将 LLM_PROVIDER 设为 openai。")
    payload = {
        "model": os.environ.get("OPENAI_MODEL", DEFAULT_LLM_MODEL),
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "reasoning": {"effort": "low"},
        "text": {"verbosity": "medium"},
        "store": False,
    }
    request = urllib.request.Request(
        os.environ.get("OPENAI_RESPONSES_URL", OPENAI_RESPONSES_URL),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            response_json = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI Responses API 请求失败（HTTP {exc.code}）：{detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 OpenAI Responses API：{exc.reason}") from exc
    output_text = response_json.get("output_text")
    if not isinstance(output_text, str) or not output_text.strip():
        raise RuntimeError("OpenAI Responses API 未返回可用 output_text。")
    return output_text.strip()


def call_llm(prompt: str, evidence_chain: dict[str, Any]) -> str:
    """LLM 统一入口：按 LLM_PROVIDER 选择真实 OpenAI 或离线模拟，并审计输出。"""
    provider = os.environ.get("LLM_PROVIDER", "mock").strip().lower()
    if provider == "openai":
        report = _call_openai_responses(prompt)
    elif provider == "mock":
        report = _mock_call_llm(prompt)
    else:
        raise ValueError("LLM_PROVIDER 仅支持 'openai' 或 'mock'。")
    _validate_llm_report(report, evidence_chain)
    return report

def agent_diagnosis_workflow(device_id: str, sensor_data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """智能体固定入口：预测 -> 检索 -> 证据链 -> Prompt -> LLM，不允许跳步或乱序。

    输入：设备 ID 与原始传感器时序；输出：Markdown 报告和完整证据链。
    """
    if not isinstance(device_id, str) or not device_id.strip():
        raise ValueError("device_id 必须为非空字符串。")
    model_input = {**sensor_data, "device_id": device_id}
    model_result = run_fault_prediction(model_input)                              # 1. 预测
    knowledge_list = retrieve_maintain_knowledge(model_result["fault_label"], 3) # 2. 检索
    evidence_chain = build_evidence_chain(model_result, knowledge_list)           # 3. 证据链
    prompt = build_report_prompt(model_result, evidence_chain, knowledge_list)    # 4. 受限 Prompt
    report = call_llm(prompt, evidence_chain)                                      # 5. LLM 整理报告并审计
    return report, evidence_chain


def _demo(name: str, device_id: str, sensor_data: dict[str, Any]) -> None:
    """运行并打印一个演示案例。"""
    report, evidence = agent_diagnosis_workflow(device_id, sensor_data)
    print(f"\n{'=' * 88}\n演示案例：{name}\n{'=' * 88}")
    print(report)
    print("\n完整可验证证据链：")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # Windows 默认 GBK 控制台可能无法编码部分 Unicode；统一以 UTF-8 输出报告。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    _demo("高置信度：轴承磨损", "PUMP-001", {
        "vibration_rms": [2.1, 3.4, 5.6, 7.2, 7.8],
        "temperature": [52.0, 61.0, 72.0, 81.0, 87.0],
        "rotational_speed": [1500, 1501, 1499, 1500, 1500],
    })
    _demo("低置信度：必须人工复核", "FAN-002", {
        "vibration_rms": [2.2, 3.1, 4.0, 4.7, 4.9],
        "temperature": [48.0, 52.0, 56.0, 60.0, 63.0],
        "rotational_speed": [1499, 1500, 1501, 1500, 1498],
    })


# 后续替换说明：
# 1) 用真实时序模型或其 API 替换 run_fault_prediction，保持返回字段与模型独立判别边界。
# 2) 用真实向量数据库替换 retrieve_maintain_knowledge，确保每个文档返回 content 和稳定 source_id。
# 3) 用真实 LLM API 替换 call_llm，并保留 Prompt 约束；调用前后校验人工复核警告、置信度、异常指标和 source_id。
