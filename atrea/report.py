"""Academic-style experiment report generation."""
from __future__ import annotations

def generate_sections(batch_id, experiments, hypothesis=None, caveats=None):
    if not experiments:
        raise ValueError("No experiments in batch")
    rows=[]
    failures=[]
    for e in experiments:
        m=e.get("metrics", {})
        rows.append(f"{e.get('model_name')}: "+", ".join(f"{k}={v}" for k,v in m.items()))
        if e.get("failure_reason") or e.get("status") not in ("completed","success",None):
            failures.append(f"{e.get('model_name')}: {e.get('failure_reason') or e.get('status')}")
    results="; ".join(rows)
    methods=("在统一数据版本、时间顺序切分、预处理流程和预测窗口下进行对照实验。"
             "所有候选模型使用相同训练/验证/测试样本与评价指标。")
    discussion=("结果仅支持当前数据集、预测跨度和运行条件下的比较；"
                "需结合多随机种子、设备/工况分层及外部数据验证泛化能力。")
    if caveats:
        discussion += " 限制："+str(caveats)
    error_notes=[]
    for e in experiments:
        diag=e.get("metrics",{}).get("error_diagnostics")
        if diag:
            error_notes.append(f"{e.get('model_name')}: MAE={diag.get('mae')}, RMSE={diag.get('rmse')}, bias={diag.get('bias')}, residual_std={diag.get('residual_std')}, maxAE={diag.get('max_absolute_error')}")
        elif e.get("error_analysis"):
            error_notes.append(f"{e.get('model_name')}: {e['error_analysis']}")
        else:
            error_notes.append(f"{e.get('model_name')}: 未提供逐样本真值/预测，无法计算残差分布。")
    return {"abstract":f"本研究围绕{hypothesis or '时间序列预测模型比较'}开展实验。共纳入{len(experiments)}项实验。主要结果为：{results}。研究结论限于当前实验协议，尚需独立数据验证。",
        "methods":methods,"results":results,"discussion":discussion,
        "error_analysis":"\n".join(error_notes),
        "failure_analysis":"\n".join(failures) if failures else "实验运行器未报告明确失败。",
        "conclusion":"仅当候选模型在冻结测试集上达到预设性能门槛、统计检验及安全约束后，才建议进入灰度发布；否则保留现有模型。"}

def render_markdown(title, sections):
    return (f"# {title}\n\n## Abstract\n{sections['abstract']}\n\n"
        f"## Methods\n{sections['methods']}\n\n## Results\n{sections['results']}\n\n"
        f"## Discussion\n{sections['discussion']}\n\n"
        f"## Error analysis\n{sections.get('error_analysis','')}\n\n"
        f"## Paper review\n{sections.get('paper_review','')}\n\n"
        f"## Prior experience review\n{sections.get('experience_review','')}\n\n"
        f"## Failure analysis\n{sections.get('failure_analysis','')}\n\n"
        f"## Conclusion\n{sections['conclusion']}\n")
