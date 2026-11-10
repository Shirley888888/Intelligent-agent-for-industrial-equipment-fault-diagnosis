"""Three-arm ablation for fixed retraining, paper-only search, and experience-enhanced search.

The host injects a runner that performs actual model fitting/evaluation on frozen splits.
No arm is allowed to bypass its own experiment protocol or fabricate scores.
"""
from __future__ import annotations
import json, uuid
from datetime import datetime, timezone
from pathlib import Path

ARMS = (
    {"id":"A1_fixed_retrain","label":"固定时间重训","trigger_policy":"fixed_interval","retrain_interval_days":1,"paper_search":False,"use_memory":False,"change_architecture":False},
    {"id":"A2_online_no_memory","label":"联网检索+架构修改（无记忆）","trigger_policy":"scheduled_research","paper_search":True,"use_memory":False,"change_architecture":True},
    {"id":"A3_experience_enhanced","label":"经验增强搜索","trigger_policy":"scheduled_research","paper_search":True,"use_memory":True,"change_architecture":True},
)

def run_three_arm_ablation(*, store, experiment_runner, paper_searcher=None,
                           query="time series forecasting foundation model",
                           dataset=None, output_dir=None):
    """Runner signature: runner(arm, papers, memory, batch_id) -> list[experiment dict]."""
    output=Path(output_dir or (store.path.parent/"ablation")); output.mkdir(parents=True,exist_ok=True)
    results=[]
    for arm in ARMS:
        batch_id=f"{arm['id']}-{uuid.uuid4().hex[:10]}"
        papers=[]
        if arm["paper_search"]:
            if paper_searcher is None:
                raise ValueError("paper_searcher is required for arms A2/A3")
            papers=paper_searcher(query,max_results=20)
        # A2 must not leave reusable paper/experiment memory behind. A3 persists its
        # reviewed source set so later A3 cycles can reuse the literature trail.
        paper_ids=([store.add_paper(p) for p in papers]
                   if arm["id"] == "A3_experience_enhanced" else [])
        memory=(store.retrieve_experience(query=query,limit=50,dataset=dataset,
                    memory_scope=arm["id"]) if arm["use_memory"] else {"experiments":[],"patterns":[]})
        records=experiment_runner(arm,papers,memory,batch_id)
        if not records:
            raise ValueError(f"{arm['id']} runner returned no experiment records")
        saved=[]
        for record in records:
            row=dict(record)
            row.update({"batch_id":batch_id,"paper_ids":paper_ids,
                        "ablation_arm":arm["id"],"created_at":datetime.now(timezone.utc).isoformat()})
            if arm["id"]=="A1_fixed_retrain" and row.get("architecture_changed"):
                raise ValueError("A1 must not change model architecture")
            if arm["change_architecture"] and not row.get("architecture_changed", False):
                raise ValueError(f"{arm['id']} must implement and record a real architecture change")
            if arm["id"]=="A2_online_no_memory" and memory["experiments"]:
                raise ValueError("A2 must not consume experience memory")
            if arm["id"]=="A3_experience_enhanced" and not row.get("experience_considered",False):
                raise ValueError("A3 runner must record that retrieved experience was considered")
            # Only A3 is allowed to write into the reusable experience memory.
            # A1/A2 results remain in arm-specific artifacts for fair ablation.
            if arm["id"] == "A3_experience_enhanced":
                row["memory_scope"] = arm["id"]
                saved.append(store.add_experiment(row))
            else:
                saved.append("artifact_only")
        pattern_ids=(store.induce_pairwise_patterns(memory_scope=arm["id"])
                     if arm["id"] == "A3_experience_enhanced" else [])
        result={"arm":arm,"batch_id":batch_id,"paper_count":len(papers),
                "memory_experiments":len(memory["experiments"]),
                "memory_patterns":len(memory["patterns"]),"experiment_ids":saved,
                "induced_pattern_ids":pattern_ids,"papers":papers,"records":records}
        # Every arm receives its own durable, paper-like experiment report.
        report_lines=[f"# {arm['label']} 实验报告", "", f"- Batch: {batch_id}",
            f"- 论文数量：{len(papers)}", f"- 规划前读取经验：{len(memory['experiments'])} 条实验、{len(memory['patterns'])} 条模式",
            f"- 归纳经验模式：{len(pattern_ids)} 条", "", "## 检索论文"]
        if papers:
            for paper in papers:
                report_lines += [f"### {paper.get('title','(untitled)')}",
                    f"- 日期：{paper.get('published','未知')}", f"- 方法线索：{', '.join(paper.get('extracted_methods', []))}",
                    f"- 摘要：{(paper.get('abstract') or '')[:1200]}", f"- 来源：{paper.get('url','')}", ""]
        else:
            report_lines += ["本组未检索论文。", ""]
        report_lines += ["## 实验结果"]
        for record in records:
            report_lines += [f"### {record.get('model_name','unknown')}",
                f"- 状态：{record.get('status','completed')}",
                f"- 指标：`{json.dumps(record.get('metrics',{}),ensure_ascii=False,default=str)}`",
                f"- 失败原因：{record.get('failure_reason') or '未报告'}",
                f"- 误差分析：{record.get('error_analysis') or '运行器未提供逐样本误差分析'}", ""]
        report_lines += ["## 方法约束与解释", f"- 架构修改要求：{arm['change_architecture']}",
            f"- 使用经验库：{arm['use_memory']}",
            "- 本报告中的性能结论仅由运行器返回的真实指标支持；未提供的结果不作推断。"]
        report_path=output/f"{arm['id']}_{batch_id}.md"
        report_path.write_text("\n".join(report_lines)+"\n",encoding="utf-8")
        result["report_path"]=str(report_path)
        results.append(result)
        # Preserve complete arm outputs, including A1/A2, without leaking them into A3 memory.
        artifact = dict(result)
        (output/f"{arm['id']}_{batch_id}.json").write_text(json.dumps(artifact,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    # Machine-readable comparison table; values come directly from runner outputs.
    metrics=sorted({k for r in results for x in r["records"] for k,v in x.get("metrics",{}).items() if isinstance(v,(int,float))})
    lines=["arm,label,experiment_count,"+",".join(metrics)]
    for r in results:
        vals=[]
        for metric in metrics:
            v=[x["metrics"][metric] for x in r["records"] if isinstance(x.get("metrics",{}).get(metric),(int,float))]
            vals.append(str(sum(v)/len(v)) if v else "")
        lines.append(",".join([r["arm"]["id"],r["arm"]["label"],str(len(r["records"])),*vals]))
    (output/"three_arm_summary.csv").write_text("\n".join(lines)+"\n",encoding="utf-8-sig")
    return {"results":results,"summary_csv":str(output/"three_arm_summary.csv")}
