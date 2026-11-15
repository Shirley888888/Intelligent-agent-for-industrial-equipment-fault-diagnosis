"""One autonomous research cycle; training/deployment functions are injected by the host project."""
from __future__ import annotations
import uuid
import numpy as np
from .papers import search_latest_papers
from .report import generate_sections,render_markdown
from .react import ReActTrace
from .research import build_research_hypothesis, diagnose_experiment
from .architecture import validate_architecture_spec
from .experience import SearchExperienceTool

def run_research_cycle(*, store, train_and_evaluate, query="time series forecasting foundation model",
                       max_papers=15, hypothesis_builder=None, batch_id=None,
                       paper_search=search_latest_papers, candidate_gate=None,
                       allow_activation=False, activate_candidate=None,
                       memory_scope="default", trace_path=None,
                       nas_train_evaluate=None, nas_ablation_evaluate=None,
                       nas_split_hash=None):

    batch_id=batch_id or str(uuid.uuid4())
    trace_path = trace_path or (store.path.parent / "react_trace.jsonl")
    trace = ReActTrace(trace_path)
    trace.log(phase="observe", action="start_research_cycle",
              observation="Research cycle started", details={"batch_id":batch_id,"query":query})
    try:
        papers=paper_search(query,max_results=max_papers,submitted_after=store.latest_paper_date())
    except TypeError:
        papers=paper_search(query,max_results=max_papers)
    paper_search_errors=getattr(papers,"provider_errors",[])
    paper_ids=[store.add_paper(p) for p in papers]
    trace.log(phase="observe", action="search_papers", observation=f"Discovered {len(papers)} candidate papers",
              artifacts=paper_ids, details={"provider_sources":sorted({p.get("source","unknown") for p in papers}),
                                            "provider_errors":paper_search_errors})
    # Experience is not injected into the planner. The planner receives a callable
    # search tool and decides whether prior papers are relevant to this research task.
    memory = {"experiments": [], "patterns": []}
    experience_tool = SearchExperienceTool(store, memory_scope=memory_scope, trace=trace)
    trace.log(phase="think", action="assess_experience_search_need",
              observation="No experience records were injected; the planner may call SearchExperience when useful",
              details={"memory_scope": memory_scope, "tool": experience_tool.name})
    # Default proposal is deliberately conservative; an LLM can replace hypothesis_builder,
    # but must return a falsifiable statement and an executable plan.
    if hypothesis_builder:
        # New planners receive both evidence sources; retain compatibility with legacy one-arg builders.
        import inspect
        try:
            signature=inspect.signature(hypothesis_builder)
            params=signature.parameters
            if "experience_tool" in params:
                kwargs={"experience_tool": experience_tool}
                if "memory" in params:
                    proposal=hypothesis_builder(papers, memory=memory, **kwargs)
                else:
                    proposal=hypothesis_builder(papers, **kwargs)
            elif len(params) >= 3:
                proposal=hypothesis_builder(papers, memory, experience_tool)
            elif len(params) >= 2:
                proposal=hypothesis_builder(papers, memory)
            else:
                proposal=hypothesis_builder(papers)
        except (TypeError, ValueError):
            proposal=hypothesis_builder(papers)
    else:
        proposal=build_research_hypothesis(papers, memory, experience_tool=experience_tool, query=query)
    if not isinstance(proposal, dict) or not str(proposal.get("statement", "")).strip():
        raise ValueError("hypothesis_builder must return a dict with a non-empty falsifiable statement")
    if not isinstance(proposal.get("plan"), dict) or not proposal["plan"]:
        raise ValueError("hypothesis_builder must return a non-empty executable plan dict")
    # Architecture proposals are validated before any host training callback is invoked.
    raw_architectures = proposal["plan"].get("candidate_architectures", [])
    if raw_architectures:
        if not isinstance(raw_architectures, list):
            raise TypeError("plan.candidate_architectures must be a list")
        proposal["plan"]["candidate_architectures"] = [
            validate_architecture_spec(spec) for spec in raw_architectures]
        trace.log(phase="think", action="validate_candidate_architectures",
                  observation=f"Validated {len(raw_architectures)} registered architecture candidate(s)",
                  details={"architectures": proposal["plan"]["candidate_architectures"]})
    # Optional real NAS execution path. The host must inject a real training
    # callback and frozen split hash; otherwise this cycle never fabricates NAS scores.
    if nas_train_evaluate is not None:
        if not nas_split_hash:
            raise ValueError("nas_split_hash is required when nas_train_evaluate is supplied")
        from .architecture import run_nas_search
        nas_cfg = proposal["plan"].get("nas_config", {})
        # Reuse prior NAS topologies as unranked starting points; all are retrained
        # on the current frozen split before they can influence selection.
        # No historical NAS records are silently injected. If the planner searched
        # the paper library, it may explicitly place a justified seed in its plan.
        prior_nas_parents = list(proposal["plan"].get("experience_nas_seeds", []))
        # De-duplicate declarative parent specs before seeding this round.
        import json as _json
        prior_nas_parents = list({_json.dumps(x, sort_keys=True): x for x in prior_nas_parents}.values())[:12]
        nas_result = run_nas_search(
            train_evaluate=nas_train_evaluate,
            ablation_evaluate=nas_ablation_evaluate,
            split_hash=nas_split_hash,
            dataset=str(proposal["plan"].get("dataset", "ETTh1")),
            population_size=int(nas_cfg.get("population_size", 12)),
            generations=int(nas_cfg.get("generations", 2)),
            seeds=list(proposal["plan"].get("seeds", [11, 22, 33])),
            metric=str(proposal["plan"].get("primary_metric", "mae")),
            parents=prior_nas_parents,
            weakness=str(proposal["plan"].get("nas_weakness")
                         or proposal.get("diagnosis", {}).get("weakness", "")))
        proposal["plan"]["nas_execution_summary"] = {
            "split_hash": nas_result["split_hash"],
            "generation_count": len(nas_result["generations"]),
            "experiment_count": len(nas_result["experiments"]),
            "ablation_count": len(nas_result["ablation_experiments"]),
            "best_candidate": nas_result["best_candidate"],
            "pareto_candidate_count": nas_result.get("pareto_candidate_count", 0),
            "pareto_candidates": nas_result.get("pareto_candidates", []),
            "next_research_hypotheses": nas_result["next_research_hypotheses"],
            "next_generation_seed_population": nas_result["next_generation_seed_population"],
            "promotion_allowed": False}
        proposal["plan"]["nas_experiments"] = nas_result["experiments"] + nas_result["ablation_experiments"]
        trace.log(phase="act", action="execute_compositional_nas",
                  observation=f"NAS completed {len(nas_result['experiments'])} candidate runs and {len(nas_result['ablation_experiments'])} ablation runs",
                  details={"generations": nas_result["generations"],
                           "best_candidate": nas_result["best_candidate"]})
    hid=store.add_hypothesis(proposal["statement"],proposal.get("rationale",""),paper_ids,[],proposal["plan"])
    trace.log(phase="think", action="form_falsifiable_hypothesis",
              observation=proposal["statement"], artifacts=[hid],
              details={"plan":proposal["plan"],"rationale":proposal.get("rationale",""),
                       "diagnosis":proposal.get("diagnosis",{})})
    trace.log(phase="act", action="train_and_evaluate", observation="Experiment runner invoked",
              details={"runner":getattr(train_and_evaluate,"__name__",type(train_and_evaluate).__name__)})
    experiments=train_and_evaluate(proposal,batch_id)
    if proposal["plan"].get("nas_experiments"):
        experiments = list(experiments or []) + list(proposal["plan"]["nas_experiments"])
    if not experiments: raise ValueError("train_and_evaluate returned no experiment records")
    for e in experiments:
        required=("dataset","split_hash","model_name","config","hyperparameters","metrics")
        missing=[k for k in required if k not in e]
        if missing: raise ValueError(f"Experiment record missing fields: {missing}")
        # Derive residual diagnostics whenever the runner returns paired truth/prediction arrays.
        truth=e.get("y_true", e.get("truth")); pred=e.get("y_pred", e.get("predictions"))
        if truth is not None and pred is not None:
            e["metrics"]["error_diagnostics"] = diagnose_experiment(e)
        elif not e.get("error_analysis") and e.get("status","completed") in ("completed","success"):
            e["error_analysis"]="运行器未提供逐样本真值/预测残差，当前只能依据汇总指标分析误差。"
    trace.log(phase="observe", action="collect_experiment_results",
              observation=f"Runner returned {len(experiments)} experiment record(s)",
              details={"statuses":{str(k):sum(1 for e in experiments if e.get("status","completed")==k)
                                    for k in sorted({e.get("status","completed") for e in experiments})}})
    gate_results=[]
    if candidate_gate:
        for candidate in experiments:
            if candidate.get("is_candidate"):
                decision=candidate_gate(candidate,experiments)
                if not isinstance(decision, dict):
                    raise TypeError("candidate_gate must return a decision dict")
                candidate["gate"] = decision
                gate_results.append({"model_name":candidate["model_name"],**decision})
    trace.log(phase="reflect", action="evaluate_candidate_gates",
              observation=f"Evaluated {len(gate_results)} candidate gate(s)", details={"gate_results":gate_results})
    ids=[]
    for e in experiments:
        e.update({"batch_id":batch_id,"paper_ids":paper_ids,"hypothesis_id":hid,
                  "memory_scope":memory_scope})
        ids.append(store.add_experiment(e))
    pattern_ids=store.induce_pairwise_patterns(memory_scope=memory_scope) if hasattr(store,"induce_pairwise_patterns") else []
    # Persist preliminary NAS insights separately from completed experiment scores.
    # They remain candidate patterns and are re-tested, never treated as proven rules.
    nas_ids = [eid for eid, row in zip(ids, experiments) if row.get("model_name") == "NAS_Composed"]
    nas_summary = proposal["plan"].get("nas_execution_summary", {})
    if hasattr(store, "add_pattern") and nas_ids:
        for hypothesis in nas_summary.get("next_research_hypotheses", []):
            pid = store.add_pattern(
                statement="NAS ablation hypothesis: " + str(hypothesis.get("statement", "retest structural component")),
                scope={"dataset": proposal["plan"].get("dataset", "ETTh1"),
                       "split_hash": nas_summary.get("split_hash"), "kind": "nas_ablation_hypothesis"},
                support_ids=nas_ids, effect={"nas_hypothesis": hypothesis},
                confidence=0.4, status="candidate", memory_scope=memory_scope)
            pattern_ids.append(pid)
        offspring = nas_summary.get("next_generation_seed_population", [])
        if offspring:
            pid = store.add_pattern(
                statement="NAS next-generation seed population; candidates require fresh training before ranking.",
                scope={"dataset": proposal["plan"].get("dataset", "ETTh1"), "kind": "nas_offspring_seed"},
                support_ids=nas_ids, effect={"next_generation_seed_population": offspring},
                confidence=0.25, status="candidate", memory_scope=memory_scope)
            pattern_ids.append(pid)
    sections=generate_sections(batch_id,experiments,proposal["statement"])
    paper_notes = [f"- {p.get('title','(untitled)')} ({p.get('published','date unknown')}): "
                   f"{(p.get('abstract') or '')[:500].replace(chr(10), ' ')} "
                   f"Methods: {', '.join(p.get('extracted_methods', []))}. Discovery score: {p.get('discovery_score','n/a')}. "
                   f"Ranking limitation: {p.get('discovery_limitations','not assessed')}. Source: {p.get('url','')}"
                   for p in papers]
    sections["paper_review"] = "\n".join(paper_notes) if paper_notes else "本轮未检索到新增论文。"
    if paper_search_errors:
        sections["paper_review"] += "\n检索源异常：" + "; ".join(
            f"{x.get('provider')}: {x.get('error')}" for x in paper_search_errors)
    sections["experience_review"] = (
        f"本轮未自动注入历史经验；Agent 自主调用经验论文搜索工具 {experience_tool.calls} 次。"
        "检索到的论文仅作为研究线索，所有方法和候选结构仍需在当前冻结切分上重新验证。"
    )
    sections["failure_analysis"] = "\n".join(
        f"- {e['model_name']}: {e.get('failure_reason') or e.get('status')}"
        for e in experiments if e.get('status','completed') not in ('completed','success') or e.get('failure_reason')
    ) or "本轮实验记录未报告明确失败。"
    if gate_results:
        sections["discussion"] += " Candidate gate outcomes: " + "; ".join(
            f"{x['model_name']}={x.get('status', x.get('allow_upgrade'))}" for x in gate_results)
    sections["conclusion"] += f" Induced {len(pattern_ids)} auditable candidate pattern(s)."
    store.add_abstract(batch_id,"Autonomous time-series experiment "+batch_id,sections,ids)
    markdown = render_markdown("Autonomous Time-Series Experiment",sections)
    report_dir = store.path.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{batch_id}.md").write_text(markdown, encoding="utf-8")
    # Store each completed research cycle as a paper in the experience library.
    exp_dataset = str(proposal["plan"].get("dataset", experiments[0].get("dataset", "未指定")))
    exp_title = f"{exp_dataset}：{proposal['statement'][:100]}"
    exp_content = "\n\n".join([
        "## 摘要\n" + str(sections.get("abstract", "")),
        "## 研究方法\n" + str(sections.get("methods", "")),
        "## 实验结果\n" + str(sections.get("results", "")),
        "## 误差与失败分析\n" + str(sections.get("failure_analysis", "")),
        "## 讨论\n" + str(sections.get("discussion", "")),
        "## 结论\n" + str(sections.get("conclusion", "")),
        "## 关联外部论文\n" + str(sections.get("paper_review", "")),
        "## 本轮研究假设\n" + str(proposal["statement"]),
    ])
    experience_paper = store.add_experience_paper(
        title=exp_title, topic=query, content=exp_content, source_batch_id=batch_id,
        dataset=exp_dataset, tags=[str(e.get("model_name", "")) for e in experiments],
        memory_scope=memory_scope)
    trace.log(phase="reflect", action="archive_experience_paper",
              observation="Archived this research cycle as a paper and refreshed the experience catalog",
              artifacts=[experience_paper["file_path"], experience_paper["catalog_path"]],
              details={"experience_id": experience_paper["experience_id"], "title": exp_title})
    trace.log(phase="reflect", action="persist_experience_and_report",
              observation=f"Persisted {len(ids)} experiments and generated research report",
              artifacts=ids+[str(report_dir / f"{batch_id}.md")],
              details={"pattern_ids":pattern_ids,"experience_used":len(memory.get("experiments",[]))})
    activation_results=[]
    if allow_activation:
        for candidate in experiments:
            decision=candidate.get("gate", {})
            if candidate.get("is_candidate") and decision.get("allow_upgrade") is True:
                required_checks=("frozen_test_passed","baseline_comparison_passed",
                                 "data_leakage_check_passed","rollback_ready")
                checks=decision.get("checks",{})
                missing=[k for k in required_checks if checks.get(k) is not True]
                if missing:
                    activation_results.append({"model_name":candidate["model_name"],
                        "status":"blocked_missing_release_checks","missing_checks":missing})
                    continue
                if activate_candidate is None:
                    activation_results.append({"model_name":candidate["model_name"],
                        "status":"blocked_no_activation_callback"})
                    continue
                # Activation runs only after the experiment, gate decision and report are durable.
                activate_candidate(candidate,decision)
                activation_results.append({"model_name":candidate["model_name"],"status":"activated"})
    trace.log(phase="act", action="candidate_activation",
              observation=f"Activation outcomes: {activation_results or 'not requested'}",
              details={"activation_enabled":allow_activation})
    return {"batch_id":batch_id,"papers_discovered":len(papers),"paper_ids":paper_ids,
            "hypothesis_id":hid,"experiment_ids":ids,"abstract":sections,
            "pattern_ids":pattern_ids,"gate_results":gate_results,
            "activation_results":activation_results,
            "markdown":markdown,"report_path":str(report_dir / f"{batch_id}.md"),
            "trace_path":str(trace_path),
            "paper_search_errors":paper_search_errors,
            "experience_used":{"experiments":0,"patterns":0,
                                "search_tool_calls":experience_tool.calls},
            "experience_paper":experience_paper}
