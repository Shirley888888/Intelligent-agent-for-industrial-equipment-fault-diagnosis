"""One autonomous research cycle; training/deployment functions are injected by the host project."""
from __future__ import annotations
import uuid
import numpy as np
from .papers import search_latest_papers
from .report import generate_sections,render_markdown
from .react import ReActTrace
from .research import build_research_hypothesis, diagnose_experiment
from .architecture import validate_architecture_spec

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
    # Retrieve the agent's own prior evidence before constructing the next hypothesis.
    memory = store.retrieve_experience(query=query, limit=30, memory_scope=memory_scope) if hasattr(store, "retrieve_experience") else {"experiments": [], "patterns": []}
    trace.log(phase="think", action="retrieve_experience",
              observation=f"Retrieved {len(memory.get('experiments', []))} experiments and {len(memory.get('patterns', []))} patterns",
              details={"memory_scope":memory_scope})
    # Default proposal is deliberately conservative; an LLM can replace hypothesis_builder,
    # but must return a falsifiable statement and an executable plan.
    if hypothesis_builder:
        # New planners receive both evidence sources; retain compatibility with legacy one-arg builders.
        import inspect
        try:
            params=inspect.signature(hypothesis_builder).parameters
            proposal=hypothesis_builder(papers, memory) if len(params) >= 2 else hypothesis_builder(papers)
        except (TypeError, ValueError):
            proposal=hypothesis_builder(papers)
    else:
        proposal=build_research_hypothesis(papers, memory)
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
        prior_nas_rows = [e for e in memory.get("experiments", [])
            if e.get("model_name") == "NAS_Composed"
            and isinstance(e.get("config"), dict)
            and e["config"].get("model_name") == "NAS_Composed"
            and e.get("dataset") == str(proposal["plan"].get("dataset", "ETTh1"))
            and e.get("status") in ("completed", "success")]
        prior_nas_rows.sort(key=lambda e: float(e.get("metrics", {}).get("mae", float("inf"))))
        prior_nas_parents = [e["config"] for e in prior_nas_rows[:4]]
        # Bring forward only explicitly marked NAS offspring/ablation hypotheses.
        for pattern in memory.get("patterns", []):
            effect = pattern.get("effect", {})
            if not isinstance(effect, dict): continue
            for candidate in effect.get("next_generation_seed_population", []):
                if isinstance(candidate, dict) and candidate.get("model_name") == "NAS_Composed":
                    prior_nas_parents.append(candidate)
            hypothesis = effect.get("nas_hypothesis", {})
            if (isinstance(hypothesis, dict)
                    and hypothesis.get("next_search_action") == "favor_simplification"):
                candidate = hypothesis.get("component_removed_spec")
                if isinstance(candidate, dict) and candidate.get("model_name") == "NAS_Composed":
                    prior_nas_parents.append(candidate)
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
    sections["experience_review"] = (f"本轮规划前检索到 {len(memory.get('experiments', []))} 条历史实验和 "
        f"{len(memory.get('patterns', []))} 条经验模式。历史经验仅作为候选依据，需在当前冻结切分上重新验证。")
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
            "experience_used":{"experiments":len(memory.get("experiments",[])),
                               "patterns":len(memory.get("patterns",[]))}}
