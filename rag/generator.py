SYSTEM_PROMPT = """你是工业设备故障知识库问答助手。
1. 只能根据提供的工业知识证据回答。
2. 禁止使用证据之外的知识推断具体故障。
3. 如果证据不足，明确回答无法判断。
4. 不得编造故障原因或维修措施。
5. 回答必须引用 KB 编号。"""

def build_prompt(query, evidence):
    evidence_text = "\n".join(f"[{x.kb_id}] {x.text}" for x in evidence) if evidence else "无有效工业知识库证据。"
    return f"""===== SYSTEM =====
{SYSTEM_PROMPT}

===== INDUSTRIAL EVIDENCE =====
{evidence_text}

===== USER QUESTION =====
{query}

===== ANSWER ====="""

def simulate_llm(query, evidence):
    if not evidence:
        return "知识库证据不足，无法判断，不编造故障结论。"
    unique = {}
    for item in evidence:
        unique[item.kb_id] = item.text
    return "".join(f"{text}[{kb_id}]" for kb_id, text in unique.items())
