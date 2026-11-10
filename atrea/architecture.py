"""Safe, declarative architecture evolution for ETTh1 forecasting.

The research agent may propose only registered architectures and whitelisted
hyperparameters. It never executes arbitrary generated Python during a cycle.
"""
from __future__ import annotations
from copy import deepcopy
import json
from typing import Any

ARCHITECTURE_CATALOG: dict[str, dict[str, Any]] = {
    "Linear": {"model_name": "Linear", "config": {}},
    "CNN1D": {"model_name": "CNN1D", "config": {"channels": 128, "kernel_size": 5}},
    "LSTM": {"model_name": "LSTM", "config": {"hidden": 128, "layers": 2}},
    "TCN": {"model_name": "TCN", "config": {"channels": [64, 64, 128], "kernel_size": 5}},
    "LSTM_Attention": {"model_name": "LSTM_Attention", "config": {"hidden": 96, "layers": 2, "heads": 4}},
    "RevIN_LSTM": {"model_name": "RevIN_LSTM", "config": {"hidden": 96, "layers": 2, "target_index": 6}},
    "PatchTransformerLite": {"model_name": "PatchTransformerLite", "config": {"d_model": 96, "patch_len": 16, "stride": 8, "heads": 4, "layers": 2}},
    "DecompLinear": {"model_name": "DecompLinear", "config": {"moving_avg": 25, "target_index": 6}},
}

# Explicit allowlists prevent a generated proposal from smuggling arbitrary
# constructor arguments into a model implementation.
_ALLOWED = {
    "Linear": set(),
    "CNN1D": {"channels", "kernel_size", "dropout"},
    "LSTM": {"hidden", "layers", "dropout"},
    "TCN": {"channels", "kernel_size", "dropout"},
    "LSTM_Attention": {"hidden", "layers", "heads", "dropout"},
    "RevIN_LSTM": {"hidden", "layers", "target_index", "dropout", "eps"},
    "PatchTransformerLite": {"d_model", "patch_len", "stride", "heads", "layers", "dropout"},
    "DecompLinear": {"moving_avg", "target_index"},
}


def validate_architecture_spec(spec: dict[str, Any], *, max_params: int = 5_000_000) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise TypeError("architecture spec must be a dict")
    name = str(spec.get("model_name", ""))
    if name == "NAS_Composed":
        from .nas import validate_nas_spec
        return validate_nas_spec(spec, max_params=max_params)
    if name not in ARCHITECTURE_CATALOG:
        raise ValueError(f"unregistered architecture: {name}")
    config = spec.get("config", {})
    if not isinstance(config, dict):
        raise TypeError("architecture config must be a dict")
    unknown = set(config) - _ALLOWED[name]
    if unknown:
        raise ValueError(f"unsupported config keys for {name}: {sorted(unknown)}")
    clean = deepcopy(config)
    for key in ("hidden", "d_model", "channels", "layers", "heads", "patch_len", "stride", "kernel_size", "moving_avg", "target_index"):
        if key in clean and key != "channels":
            value = clean[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
    if "channels" in clean:
        channels = clean["channels"]
        if name == "CNN1D":
            if not isinstance(channels, int) or not 1 <= channels <= 1024:
                raise ValueError("CNN1D channels must be an integer in [1, 1024]")
        else:
            if isinstance(channels, int):
                channels = [channels]
            if not isinstance(channels, (list, tuple)) or not channels or any(not isinstance(v, int) or v < 1 or v > 1024 for v in channels):
                raise ValueError("TCN channels must be a non-empty list of integers in [1, 1024]")
            clean["channels"] = list(channels)
    for key in ("hidden", "d_model", "layers", "heads", "patch_len", "stride", "kernel_size", "moving_avg"):
        if key in clean and clean[key] < 1:
            raise ValueError(f"{key} must be positive")
    for key in ("dropout",):
        if key in clean and not 0 <= float(clean[key]) < 1:
            raise ValueError("dropout must be in [0, 1)")
    if name in ("LSTM_Attention", "PatchTransformerLite"):
        width = clean.get("hidden" if name == "LSTM_Attention" else "d_model", 96)
        heads = clean.get("heads", 4)
        if width % heads:
            raise ValueError("model width must be divisible by attention heads")
    if name in ("RevIN_LSTM", "DecompLinear") and clean.get("target_index", 6) not in range(7):
        raise ValueError("target_index must be in [0, 6] for ETTh1")
    # Conservative static estimate used before model allocation; actual count is
    # checked again after instantiation.
    width = int(clean.get("hidden", clean.get("d_model", 128)))
    if width * width * max(1, int(clean.get("layers", 2))) > max_params:
        raise ValueError("architecture exceeds configured complexity budget")
    return {"model_name": name, "config": clean}


def build_model(spec: dict[str, Any], *, input_len: int = 96, n_features: int = 7,
                pred_len: int = 24, max_params: int = 5_000_000):
    """Instantiate a registered model and enforce parameter budget."""
    clean = validate_architecture_spec(spec, max_params=max_params)
    from models import MODEL_CLASSES
    model = MODEL_CLASSES[clean["model_name"]](input_len=input_len, n_features=n_features,
                                                pred_len=pred_len, **clean["config"])
    count = sum(p.numel() for p in model.parameters())
    if count > max_params:
        raise ValueError(f"model has {count} parameters; budget is {max_params}")
    return model


def propose_architectures(methods: list[str], weakness: str = "") -> list[dict[str, Any]]:
    """Map evidence-backed paper method leads to runnable architecture specs."""
    candidates: list[str] = []
    method_set = {str(m).lower() for m in methods}
    weak = weakness.lower()
    if "normalization" in method_set or "bias" in weak or "scale" in weak:
        candidates.append("RevIN_LSTM")
    if "patching" in method_set or "long" in weak or "后段" in weakness:
        candidates.append("PatchTransformerLite")
    if "decomposition" in method_set or "trend" in weak or "周期" in weakness:
        candidates.append("DecompLinear")
    if "frequency_modeling" in method_set:
        candidates.extend(["PatchTransformerLite", "DecompLinear"])
    if "attention" in method_set or "long" in weak or "后段" in weakness:
        candidates.append("LSTM_Attention")
    # Always include a structurally meaningful hybrid candidate if literature
    # retrieval is sparse, while retaining a plain incumbent comparison.
    if not candidates:
        candidates = ["LSTM_Attention", "PatchTransformerLite"]
    result = []
    for name in dict.fromkeys(candidates):
        result.append(validate_architecture_spec(ARCHITECTURE_CATALOG[name]))
    return result[:4]


def describe_architecture_space() -> dict[str, Any]:
    return {"registered_architectures": sorted(ARCHITECTURE_CATALOG) + ["NAS_Composed"],
            "nas_search_dimensions": ["branch_topology", "layer_depth", "hidden_width",
                "registered_operator_composition", "meta_operator_primitive_graph", "mathematical_equation_discovery", "equation_path_search", "primitive_order_and_depth", "attention_placement", "feature_fusion",
                "loss_function", "optimizer", "learning_rate", "weight_decay", "scheduler"],
            "safe_generation_mode": "declarative_registered_components_only",
            "arbitrary_python_execution": False,
            "parameter_budget_default": 5_000_000}


def run_architecture_sweep(specs: list[dict[str, Any]], train_evaluate, *,
                           dataset: str = "ETTh1", split_hash: str,
                           seeds: list[int] | None = None) -> list[dict[str, Any]]:
    """Execute registered candidates through a host's real training/evaluation callback.

    The callback contract is ``train_evaluate(spec, seed) -> metrics record``.
    Each candidate is isolated: one failure does not prevent the remaining runs.
    This helper does not fabricate metrics or silently mark a failed run successful.
    """
    if not callable(train_evaluate):
        raise TypeError("train_evaluate must be callable")
    if not split_hash:
        raise ValueError("split_hash is required for comparable experiments")
    seeds = list(seeds or [11, 22, 33])
    if not seeds or any(not isinstance(s, int) for s in seeds):
        raise ValueError("seeds must be a non-empty list of integers")
    records = []
    for raw in specs:
        try:
            spec = validate_architecture_spec(raw)
        except Exception as exc:
            records.append({"dataset": dataset, "split_hash": split_hash,
                "model_name": str(raw.get("model_name", "invalid")) if isinstance(raw, dict) else "invalid",
                "config": raw.get("config", {}) if isinstance(raw, dict) else {},
                "hyperparameters": {}, "metrics": {}, "status": "rejected",
                "is_candidate": True, "failure_reason": f"architecture_validation: {exc}"})
            continue
        for seed in seeds:
            try:
                result = train_evaluate(spec, seed)
                if not isinstance(result, dict) or not isinstance(result.get("metrics"), dict):
                    raise ValueError("callback must return a dict containing metrics dict")
                records.append({"dataset": dataset, "split_hash": split_hash,
                    "model_name": spec["model_name"], "config": spec,
                    "hyperparameters": {**spec["config"], "seed": seed},
                    "metrics": result["metrics"], "status": result.get("status", "completed"),
                    "is_candidate": True, "seed": seed,
                    **{k: v for k, v in result.items() if k not in ("metrics", "status")}})
            except Exception as exc:
                records.append({"dataset": dataset, "split_hash": split_hash,
                    "model_name": spec["model_name"], "config": spec,
                    "hyperparameters": {**spec["config"], "seed": seed},
                    "metrics": {}, "status": "failed", "is_candidate": True,
                    "seed": seed, "failure_reason": f"training_or_evaluation: {type(exc).__name__}: {exc}"})
    return records


def run_nas_search(*, train_evaluate, split_hash: str, dataset: str = "ETTh1",
                   population_size: int = 12, generations: int = 2,
                   seeds: list[int] | None = None, seed: int = 2026,
                   parents: list[dict[str, Any]] | None = None,
                   metric: str = "mae", ablation_evaluate=None,
                   max_params: int = 5_000_000, weakness: str = "") -> dict[str, Any]:
    """Run bounded evolutionary NAS using a host's real fit/evaluate callback.

    train_evaluate(spec, seed, loss_fn) must return {"metrics": {...}}.
    ablation_evaluate(spec, seed, loss_fn) is optional and is called for the
    best candidate's leave-one-component-out variants. No test set is used here.
    """
    from .nas import (generate_nas_population, build_loss, make_ablation_specs,
                      rank_experiments, pareto_front, propose_operator_innovations)
    if not callable(train_evaluate) or not split_hash:
        raise ValueError("a real train_evaluate callback and frozen split_hash are required")
    if not 1 <= generations <= 10:
        raise ValueError("generations must be in [1,10]")
    seeds = list(seeds or [11, 22, 33])
    if not seeds or any(not isinstance(x, int) for x in seeds):
        raise ValueError("seeds must be non-empty integers")
    population = list(parents or [])
    all_records=[]; generation_rows=[]
    for gen in range(generations):
        specs = generate_nas_population(parents=population or None,
                    population_size=population_size, seed=seed+gen,
                    max_params=max_params)
        # In generation zero, reserve slots for explicit error-driven operator hypotheses.
        if gen == 0 and weakness:
            innovations = propose_operator_innovations(weakness=weakness, seed=seed,
                parent_spec=(population[0] if population else None))
            known = {json.dumps(x, sort_keys=True, default=str) for x in specs}
            replacement_index = len(specs) - 1
            for candidate in innovations:
                clean_candidate = validate_architecture_spec(candidate, max_params=max_params)
                key = json.dumps(clean_candidate, sort_keys=True, default=str)
                if key not in known and replacement_index >= 0:
                    clean_candidate["proposal"] = candidate.get("proposal")
                    specs[replacement_index] = clean_candidate
                    replacement_index -= 1
                    known.add(key)
        gen_records=[]
        for spec in specs:
            clean=validate_architecture_spec(spec,max_params=max_params)
            loss_name=clean["config"]["loss"]
            loss_fn=build_loss(loss_name, loss_alpha=clean["config"].get("loss_alpha", .25))
            for run_seed in seeds:
                try:
                    result=train_evaluate(clean,run_seed,loss_fn)
                    if not isinstance(result,dict) or not isinstance(result.get("metrics"),dict):
                        raise ValueError("NAS runner must return {'metrics': {...}}")
                    row={"dataset":dataset,"split_hash":split_hash,
                         "model_name":"NAS_Composed","config":clean,
                         "architecture_id":spec.get("architecture_id"),
                         "hyperparameters":{"seed":run_seed,**clean["config"]},
                         "seed":run_seed,"loss_name":loss_name,
                         "metrics":result["metrics"],
                         "status":result.get("status","completed"),
                         "operator_innovation":spec.get("proposal"),
                         "is_candidate":True,"nas_generation":gen,
                         "ablation_role":"parent_candidate",
                         **{k:v for k,v in result.items() if k not in ("metrics","status")}}
                except Exception as exc:
                    row={"dataset":dataset,"split_hash":split_hash,
                         "model_name":"NAS_Composed","config":clean,
                         "architecture_id":spec.get("architecture_id"),
                         "hyperparameters":{"seed":run_seed,**clean["config"]},
                         "seed":run_seed,"loss_name":loss_name,"metrics":{},
                         "status":"failed","operator_innovation":spec.get("proposal"),
                         "is_candidate":True,"nas_generation":gen,
                         "failure_reason":f"{type(exc).__name__}: {exc}"}
                gen_records.append(row); all_records.append(row)
        raw_ranked=rank_experiments(gen_records,metric=metric)
        # Rank architecture means across seeds, not the luckiest individual seed.
        grouped={}
        for row in raw_ranked:
            key=json.dumps(row["config"],sort_keys=True,default=str)
            grouped.setdefault(key,[]).append(row)
        ranked=[]
        for rows in grouped.values():
            vals=[float(x["metrics"][metric]) for x in rows]
            import statistics
            aggregate=dict(rows[0])
            aggregate["metrics"]=dict(rows[0]["metrics"])
            metric_keys=set.intersection(*(set(x.get("metrics",{})) for x in rows)) if rows else set()
            for mk in metric_keys:
                values=[float(x["metrics"][mk]) for x in rows if isinstance(x.get("metrics",{}).get(mk),(int,float))]
                if values: aggregate["metrics"][mk]=sum(values)/len(values)
            aggregate["metrics"][metric]=sum(vals)/len(vals)
            aggregate["seed_metric_std"]=statistics.pstdev(vals) if len(vals)>1 else 0.0
            aggregate["seed_runs"]=len(vals)
            for key in ("parameter_count", "train_seconds"):
                values=[float(x[key]) for x in rows if isinstance(x.get(key),(int,float))]
                if values: aggregate[key]=sum(values)/len(values)
            ranked.append(aggregate)
        ranked.sort(key=lambda x:float(x["metrics"][metric]))
        # Preserve diversity: select the Pareto front first, then fill by primary metric.
        front = pareto_front(ranked)
        survivor_rows = list(front)
        for row in ranked:
            if row not in survivor_rows:
                survivor_rows.append(row)
            if len(survivor_rows) >= max(1,min(4,len(ranked))): break
        population=[row["config"] for row in survivor_rows[:max(1,min(4,len(ranked)))]]
        generation_rows.append({"generation":gen,"candidate_count":len(specs),
             "successful_runs":len(raw_ranked),"architectures_ranked":len(ranked),
             "best_metric":ranked[0]["metrics"][metric] if ranked else None,
             "survivor_count":len(population)})
        if not population: break
    raw_all=rank_experiments(all_records,metric=metric)
    grouped_all={}
    for row in raw_all:
        key=json.dumps(row["config"],sort_keys=True,default=str)
        grouped_all.setdefault(key,[]).append(row)
    ranked_all=[]
    for rows in grouped_all.values():
        vals=[float(x["metrics"][metric]) for x in rows]
        import statistics
        aggregate=dict(rows[0]); aggregate["metrics"]=dict(rows[0]["metrics"])
        metric_keys=set.intersection(*(set(x.get("metrics",{})) for x in rows)) if rows else set()
        for mk in metric_keys:
            values=[float(x["metrics"][mk]) for x in rows if isinstance(x.get("metrics",{}).get(mk),(int,float))]
            if values: aggregate["metrics"][mk]=sum(values)/len(values)
        aggregate["metrics"][metric]=sum(vals)/len(vals)
        aggregate["seed_metric_std"]=statistics.pstdev(vals) if len(vals)>1 else 0.0
        aggregate["seed_runs"]=len(vals)
        for key in ("parameter_count", "train_seconds"):
            values=[float(x[key]) for x in rows if isinstance(x.get(key),(int,float))]
            if values: aggregate[key]=sum(values)/len(values)
        ranked_all.append(aggregate)
    ranked_all.sort(key=lambda x:float(x["metrics"][metric]))
    pareto_candidates = pareto_front(ranked_all)
    best=ranked_all[0] if ranked_all else None
    ablation_records=[]
    if best and callable(ablation_evaluate):
        parent_fp = __import__("atrea.nas", fromlist=["_fingerprint"])._fingerprint(best["config"])
        parent_by_seed = {r["seed"]: float(r["metrics"][metric]) for r in all_records
                          if r.get("status") in ("completed", "success")
                          and r.get("config") == best["config"]
                          and isinstance(r.get("metrics", {}).get(metric), (int, float))}
        for ab_spec in make_ablation_specs(best["config"]):
            for run_seed in seeds:
                try:
                    result=ablation_evaluate(ab_spec,run_seed,build_loss(ab_spec["config"]["loss"], loss_alpha=ab_spec["config"].get("loss_alpha", .25)))
                    if not isinstance(result,dict) or not isinstance(result.get("metrics"),dict):
                        raise ValueError("ablation callback must return metrics")
                    ablation_records.append({"dataset":dataset,"split_hash":split_hash,
                        "model_name":"NAS_Composed","config":ab_spec,"seed":run_seed,
                        "hyperparameters":{"seed":run_seed,**ab_spec["config"]},
                        "loss_name":ab_spec["config"]["loss"],"metrics":result["metrics"],
                        "status":result.get("status","completed"),"is_candidate":True,
                        "ablation_role":"component_ablation",
                        "parent_architecture_id":parent_fp,
                        "parent_metric":parent_by_seed.get(run_seed)})
                except Exception as exc:
                    ablation_records.append({"dataset":dataset,"split_hash":split_hash,
                        "model_name":"NAS_Composed","config":ab_spec,"seed":run_seed,
                        "hyperparameters":{"seed":run_seed,**ab_spec["config"]},
                        "loss_name":ab_spec["config"]["loss"],"metrics":{},
                        "status":"failed","is_candidate":True,"ablation_role":"component_ablation",
                        "failure_reason":f"{type(exc).__name__}: {exc}",
                        "parent_architecture_id":parent_fp,
                        "parent_metric":parent_by_seed.get(run_seed)})
    component_effects=[]
    if best and ablation_records:
        base_value=float(best["metrics"][metric])
        by_config={}
        for row in ablation_records:
            if row.get("status") in ("completed","success") and isinstance(row.get("metrics",{}).get(metric),(int,float)):
                key=json.dumps(row["config"],sort_keys=True,default=str)
                by_config.setdefault(key,[]).append(float(row["metrics"][metric]))
        for key,vals in by_config.items():
            ab_mean=sum(vals)/len(vals)
            delta=ab_mean-base_value
            component_effects.append({"ablation_config":json.loads(key),
                "baseline_metric":base_value,"ablation_metric_mean":ab_mean,
                "delta_vs_parent":delta,
                "interpretation":"component_may_be_useful" if delta>0 else "candidate_for_removal",
                "seed_runs":len(vals)})
    from .nas import infer_structure_hypotheses, generate_nas_population
    next_hypotheses = infer_structure_hypotheses(best["config"], ablation_records, metric=metric) if best else []
    next_population = generate_nas_population(parents=[best["config"]],
        population_size=min(12, max(1, population_size)), seed=seed + generations + 1000,
        max_params=max_params) if best else []
    return {"search_mode":"bilevel_differentiable_symbolic_nas_v9","metric":metric,
            "objectives":["mae","rmse","parameter_count","train_seconds"],
            "pareto_candidate_count":len(pareto_candidates),
            "pareto_candidates":pareto_candidates,
            "split_hash":split_hash,"generations":generation_rows,
            "experiments":all_records,"ranked_candidates":ranked_all,
            "best_candidate":best,"ablation_experiments":ablation_records,
            "ablation_count":len(ablation_records),
            "component_effects":component_effects,
            "next_research_hypotheses":next_hypotheses,
            "next_generation_seed_population":next_population,
            "promotion_allowed":False,
            "note":"Validation ranking only; final untouched test and release gates remain mandatory."}
