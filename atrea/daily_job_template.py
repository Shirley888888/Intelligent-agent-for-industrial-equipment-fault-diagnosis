"""Environment-configured daily research callback for atrea.scheduler.

Set ATREA_EXPERIMENT_RUNNER to a Python factory (module:function) returning a
callable runner(proposal, batch_id). For the included Chronos-2 integration use:
ATREA_EXPERIMENT_RUNNER=atrea.chronos2_runner:build_runner_from_env
"""
from __future__ import annotations
import importlib
import os
from .cycle import run_research_cycle
from .store import ExperienceStore


def _load_factory(spec):
    module_name, function_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), function_name)


def run_daily_job(now=None):
    runner_spec = os.getenv("ATREA_EXPERIMENT_RUNNER")
    if not runner_spec:
        raise RuntimeError("Set ATREA_EXPERIMENT_RUNNER=module:function before starting the daily scheduler")
    runner = _load_factory(runner_spec)()
    builder_spec = os.getenv("ATREA_HYPOTHESIS_BUILDER")
    if not builder_spec:
        raise RuntimeError("Set ATREA_HYPOTHESIS_BUILDER to a factory returning hypothesis_builder(papers, memory, experience_tool); unattended research requires a configured planner. The planner receives an empty memory object and must call experience_tool.search_experience(query) when it judges prior experience relevant")
    hypothesis_builder = _load_factory(builder_spec)()
    workspace = os.getenv("ATREA_WORKSPACE", "agent_workspace/atrea_experience.sqlite3")
    query = os.getenv("ATREA_PAPER_QUERY", "industrial transformer oil temperature time series forecasting foundation model")
    store = ExperienceStore(workspace)
    result = run_research_cycle(store=store, train_and_evaluate=runner,
        query=query, hypothesis_builder=hypothesis_builder,
        memory_scope=os.getenv("ATREA_MEMORY_SCOPE", "default"),
        trace_path=os.getenv("ATREA_REACT_TRACE", "agent_workspace/react_trace.jsonl"))
    return {"batch_id":result["batch_id"], "papers_discovered":result["papers_discovered"],
            "experiment_ids":result["experiment_ids"], "report_path":result["report_path"],
            "trace_path":result["trace_path"]}
