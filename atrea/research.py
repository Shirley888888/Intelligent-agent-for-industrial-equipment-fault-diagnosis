"""Evidence-driven research planning for ATREA.

This module deliberately separates paper discovery from empirical claims: paper
metadata proposes candidate methods, while only executed experiments establish
whether a method helps on the current task.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any


def diagnose_experiment(record: dict[str, Any]) -> dict[str, Any]:
    """Create reproducible residual diagnostics from aligned truth/prediction arrays."""
    import numpy as np

    truth = record.get("y_true", record.get("truth"))
    pred = record.get("y_pred", record.get("predictions"))
    if truth is None or pred is None:
        return {"available": False, "reason": "runner_did_not_return_aligned_truth_and_prediction"}
    y = np.asarray(truth, dtype=float)
    p = np.asarray(pred, dtype=float)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("truth and prediction must have identical non-empty shapes")
    err = p - y
    abs_err = np.abs(err)
    result = {
        "available": True,
        "shape": list(y.shape),
        "n": int(y.size),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(err.mean()),
        "residual_std": float(err.std()),
        "max_absolute_error": float(abs_err.max()),
        "p95_absolute_error": float(np.quantile(abs_err, .95)),
    }
    # Common forecasting layout is [sample, horizon, target] or [sample, horizon].
    if y.ndim >= 2:
        horizon_axis = 1
        reduce_axes = tuple(i for i in range(y.ndim) if i != horizon_axis)
        result["mae_by_horizon"] = np.mean(abs_err, axis=reduce_axes).astype(float).tolist()
        result["bias_by_horizon"] = np.mean(err, axis=reduce_axes).astype(float).tolist()
        h_mae = np.asarray(result["mae_by_horizon"])
        result["worst_horizon"] = int(np.argmax(h_mae)) + 1
        result["best_horizon"] = int(np.argmin(h_mae)) + 1
        result["late_horizon_error_ratio"] = float(
            (h_mae[max(0, len(h_mae) * 2 // 3):].mean() + 1e-12) /
            (h_mae[:max(1, len(h_mae) // 3)].mean() + 1e-12))
    return result


def _method_candidates(papers: list[dict[str, Any]]) -> list[str]:
    methods = []
    for paper in papers:
        methods.extend(str(x).lower() for x in paper.get("extracted_methods", []))
        text = (str(paper.get("title", "")) + " " + str(paper.get("abstract", ""))).lower()
        for method, patterns in {
            "patching": (r"patch[- ]based", r"patchtst", r"patching"),
            "decomposition": (r"decomposition", r"trend[- ]seasonal"),
            "channel_independence": (r"channel[- ]independent", r"channel independence"),
            "probabilistic_forecasting": (r"probabilistic forecasting", r"quantile forecast"),
            "normalization": (r"reversible instance normalization", r"revin"),
            "frequency_modeling": (r"frequency domain", r"fourier", r"frequency-aware"),
        }.items():
            if any(re.search(p, text) for p in patterns):
                methods.append(method)
    # Stable order makes plans deterministic and auditable.
    return list(dict.fromkeys(methods))


def build_research_hypothesis(papers: list[dict[str, Any]], memory: dict[str, Any]) -> dict[str, Any]:
    """Build a falsifiable, conservative plan from residual evidence and paper leads."""
    experiments = memory.get("experiments", [])
    failed = [e for e in experiments if e.get("status") not in ("completed", "success")]
    recent = experiments[:10]
    diagnostics = [e.get("metrics", {}).get("error_diagnostics", {}) for e in recent]
    diagnostics = [d for d in diagnostics if d]
    methods = _method_candidates(papers)
    # Prior failed candidates are explicitly excluded from automatic repetition.
    failed_names = {e.get("model_name", "") for e in failed}
    if diagnostics:
        avg_bias = sum(float(d.get("bias", 0.0)) for d in diagnostics) / len(diagnostics)
        ratios = [float(d.get("late_horizon_error_ratio", 1.0)) for d in diagnostics
                  if d.get("late_horizon_error_ratio") is not None]
        if ratios and sum(ratios) / len(ratios) > 1.15:
            weakness = "预测后段误差明显增大，提示长预测跨度上的误差累积"
            preferred = [m for m in methods if m in ("patching", "decomposition", "frequency_modeling")]
        elif abs(avg_bias) > 0.05:
            weakness = f"历史残差存在系统性偏差（平均 bias={avg_bias:.4f}）"
            preferred = [m for m in methods if m in ("normalization", "decomposition")]
        else:
            weakness = "现有残差证据尚未显示单一主导失效模式，需要进行结构化候选比较"
            preferred = methods
    else:
        weakness = "历史实验尚缺少逐样本残差或可分解指标，首轮应先补齐误差诊断证据"
        preferred = methods
    if not preferred:
        preferred = methods[:2]
    # Do not fabricate a method or claim a paper proves improvement.
    candidate_methods = preferred[:3]
    # Translate paper method leads and observed weaknesses into executable,
    # whitelisted architecture specifications (not merely method labels).
    from .architecture import propose_architectures
    architecture_candidates = propose_architectures(candidate_methods, weakness)
    # Add a bounded, declarative NAS seed population: topology is now searchable,
    # not restricted to choosing one of the fixed catalog entries.
    from .nas import seed_architecture_population
    nas_seed_population = seed_architecture_population(seed=2026)
    rationale = (f"诊断结论：{weakness}。论文检索仅提供候选方法线索；"
                 "所有方法必须在相同时间切分、预处理和随机种子下重新验证。")
    return {
        "statement": f"针对{weakness}，检验论文候选方法是否能在 ETTh1/当前任务冻结验证协议上改善 MAE，且不恶化其他关键指标。",
        "rationale": rationale,
        "diagnosis": {"weakness": weakness, "candidate_methods": candidate_methods,
                      "historical_failed_models": sorted(x for x in failed_names if x),
                      "candidate_architectures": architecture_candidates,
                      "nas_seed_population": nas_seed_population},
        "plan": {
            "protocol": "paired_same_split_same_preprocessing",
            "required_groups": ["incumbent_baseline", "candidate_method", "ablation_if_applicable"],
            "candidate_methods": candidate_methods,
            "candidate_architectures": architecture_candidates,
            "nas_seed_population": nas_seed_population,
            "architecture_search_mode": "evolutionary_compositional_nas",
            "nas_config": {"population_size": 12, "generations": 2,
                           "search_dimensions": ["block_topology", "depth", "width",
                             "trend_residual_operator", "frequency_mixer_operator",
                             "cross_variable_interaction", "attention_placement",
                             "feature_fusion", "loss_function", "optimizer",
                             "learning_rate", "weight_decay", "scheduler"],
                           "ablation": "leave_one_component_out"},
            "seeds": [11, 22, 33],
            "primary_metric": "mae",
            "secondary_metrics": ["rmse", "bias", "p95_absolute_error"],
            "require_residuals": True,
            "require_data_leakage_check": True,
            "do_not_repeat_failed_models": sorted(x for x in failed_names if x),
            "paper_evidence_is_not_performance_evidence": True,
        },
    }
