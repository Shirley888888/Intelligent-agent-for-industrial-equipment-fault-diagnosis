"""Trigger-gated autonomous research orchestration.

The host supplies fresh observations, matured labels, and a safe experiment runner.
Candidate activation is opt-in and must be implemented by the host's release gate.
"""
from __future__ import annotations

from dataclasses import asdict
from .cycle import run_research_cycle
from .trigger import TriggerConfig, evaluate_training_trigger


def run_if_due(*, store, errors, reference_features=None, recent_features=None,
               last_train_time=None, now=None, consecutive_bad_checks=0,
               config: TriggerConfig | None=None, experiment_runner,
               query="time series forecasting foundation model", max_papers=15,
               paper_search=None, hypothesis_builder=None, candidate_gate=None,
               allow_activation=False, activate_candidate=None, memory_scope="default",
               trace_path=None):
    """Evaluate current data and run one experiment cycle only when triggered."""
    decision=evaluate_training_trigger(errors=errors,
        reference_features=reference_features,recent_features=recent_features,
        last_train_time=last_train_time,now=now,
        consecutive_bad_checks=consecutive_bad_checks,config=config)
    result={"trigger":asdict(decision),"cycle":None}
    if not decision.should_train:
        return result
    kwargs={"store":store,"train_and_evaluate":experiment_runner,
            "query":query,"max_papers":max_papers,
            "hypothesis_builder":hypothesis_builder,
            "candidate_gate":candidate_gate,
            "allow_activation":allow_activation,
            "activate_candidate":activate_candidate,
            "memory_scope":memory_scope,
            "trace_path":trace_path}
    if paper_search is not None: kwargs["paper_search"]=paper_search
    result["cycle"]=run_research_cycle(**kwargs)
    return result
