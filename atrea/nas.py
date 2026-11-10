"""Autonomous, bounded neural architecture search for time-series forecasting.

Searches topology, depth, attention placement, fusion and loss recipes using
declarative specs only. No generated Python is imported or executed.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib, json, random
from typing import Any

BLOCKS = ("lstm", "cnn", "transformer", "linear", "trend_residual", "frequency_mixer", "cross_var", "amplitude_phase")
META_PRIMITIVES = ("layer_norm", "depthwise_conv", "pointwise_mlp", "gated_mlp",
                   "temporal_attention", "spectral_gain", "trend_split", "channel_gate")
# V7: bounded mathematical equation templates. These are explicit, testable
# function families, not arbitrary generated Python or free-form expressions.
MATH_EQUATIONS = ("residual_add", "gated_residual", "multiplicative_modulation",
                  "parallel_residual", "trend_spectral_mix")
DARTS_PRIMITIVE_LIMIT = 8
ATTENTION = ("none", "input", "after_each", "branch_end", "fusion")
FUSION = ("concat", "mean", "gated")
LOSSES = ("mae", "huber", "mae_rmse", "horizon_weighted_mae", "mae_slope")
OPTIMIZERS = ("adamw", "adam", "rmsprop")
SCHEDULERS = ("none", "cosine", "plateau")

def validate_nas_spec(spec: dict[str, Any], *, max_params: int = 5_000_000) -> dict[str, Any]:
    if not isinstance(spec, dict) or spec.get("model_name") != "NAS_Composed":
        raise ValueError("NAS spec must use model_name=NAS_Composed")
    c = deepcopy(spec.get("config", {}))
    allowed = {"branches", "hidden", "attention", "attention_heads", "fusion", "dropout", "loss",
               "optimizer", "learning_rate", "weight_decay", "scheduler", "loss_alpha"}
    if set(c) - allowed:
        raise ValueError(f"unknown NAS config keys: {sorted(set(c)-allowed)}")
    branches = c.get("branches")
    if not isinstance(branches, list) or not 1 <= len(branches) <= 3:
        raise ValueError("branches must contain 1..3 parallel branches")
    for branch in branches:
        if not isinstance(branch, list) or not 1 <= len(branch) <= 5:
            raise ValueError("each branch must contain 1..5 registered blocks")
        for block in branch:
            if isinstance(block, str):
                if block not in BLOCKS: raise ValueError(f"unregistered block: {block}")
            elif isinstance(block, dict) and block.get("type") == "meta_operator" and set(block) == {"type", "ops"}:
                ops=block.get("ops")
                if not isinstance(ops,list) or not 1 <= len(ops) <= 6 or any(op not in META_PRIMITIVES for op in ops):
                    raise ValueError("meta_operator ops must be 1..6 registered primitives")
            elif isinstance(block, dict) and block.get("type") == "darts_operator" and set(block) == {"type", "ops"}:
                ops = block.get("ops")
                if not isinstance(ops, list) or not 2 <= len(ops) <= DARTS_PRIMITIVE_LIMIT or len(set(ops)) != len(ops) or any(op not in META_PRIMITIVES for op in ops):
                    raise ValueError("darts_operator ops must contain 2..8 unique registered primitives")
            elif isinstance(block, dict) and block.get("type") == "math_operator" and set(block) == {"type", "equation", "ops", "gate_ops"}:
                if block.get("equation") not in MATH_EQUATIONS:
                    raise ValueError("math_operator equation is not registered")
                ops, gate_ops = block.get("ops"), block.get("gate_ops")
                if not isinstance(ops,list) or not 1 <= len(ops) <= 6 or any(op not in META_PRIMITIVES for op in ops):
                    raise ValueError("math_operator ops must be 1..6 registered primitives")
                if not isinstance(gate_ops,list) or len(gate_ops)>4 or any(op not in META_PRIMITIVES for op in gate_ops):
                    raise ValueError("math_operator gate_ops must contain 0..4 registered primitives")
                if block["equation"] in ("gated_residual", "multiplicative_modulation", "parallel_residual") and not gate_ops:
                    raise ValueError("selected equation requires a non-empty gate_ops path")
            else:
                raise ValueError("blocks must be registered strings or declarative operator graphs")
    hidden = c.get("hidden", 64); heads = c.get("attention_heads", 4)
    if not isinstance(hidden, int) or not 16 <= hidden <= 256:
        raise ValueError("hidden must be an integer in [16,256]")
    if not isinstance(heads, int) or heads < 1 or hidden % heads:
        raise ValueError("attention_heads must divide hidden")
    if c.get("attention", "none") not in ATTENTION:
        raise ValueError("unsupported attention placement")
    if c.get("fusion", "concat") not in FUSION:
        raise ValueError("unsupported fusion")
    if c.get("loss", "mae") not in LOSSES:
        raise ValueError("unsupported loss")
    if c.get("optimizer", "adamw") not in OPTIMIZERS:
        raise ValueError("unsupported optimizer")
    if c.get("scheduler", "none") not in SCHEDULERS:
        raise ValueError("unsupported scheduler")
    lr=float(c.get("learning_rate", 1e-3)); wd=float(c.get("weight_decay", 1e-4))
    alpha=float(c.get("loss_alpha", .25))
    if not 1e-5 <= lr <= 5e-2: raise ValueError("learning_rate must be in [1e-5,5e-2]")
    if not 0 <= wd <= .1: raise ValueError("weight_decay must be in [0,.1]")
    if not 0 <= alpha <= 2: raise ValueError("loss_alpha must be in [0,2]")
    dropout = float(c.get("dropout", .1))
    if not 0 <= dropout < .8:
        raise ValueError("dropout must be in [0,.8)")
    # Conservative topology budget; build_model checks actual parameter count.
    blocks = sum(map(len, branches))
    if hidden * hidden * blocks * 12 > max_params:
        raise ValueError("NAS topology exceeds static complexity budget")
    c.update({"hidden": hidden, "attention_heads": heads, "dropout": dropout,
              "optimizer": c.get("optimizer", "adamw"), "learning_rate": lr,
              "weight_decay": wd, "scheduler": c.get("scheduler", "none"),
              "loss_alpha": alpha})
    return {"model_name": "NAS_Composed", "config": c}

def _fingerprint(spec):
    raw=json.dumps(spec,sort_keys=True,separators=(",",":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def seed_architecture_population(seed: int = 2026) -> list[dict[str, Any]]:
    rng=random.Random(seed)
    base=[
      {"branches":[["lstm","lstm"]],"hidden":64,"attention":"none","fusion":"concat","loss":"mae"},
      {"branches":[["trend_residual","lstm"]],"hidden":64,"attention":"branch_end","fusion":"concat","loss":"mae_slope"},
      {"branches":[["frequency_mixer","transformer"]],"hidden":64,"attention":"none","fusion":"mean","loss":"huber"},
      {"branches":[["cross_var","lstm"],["cnn","trend_residual"]],"hidden":64,"attention":"fusion","fusion":"gated","loss":"mae_rmse"},
      {"branches":[["cnn","lstm"]],"hidden":64,"attention":"branch_end","fusion":"concat","loss":"huber"},
      {"branches":[["lstm","transformer"]],"hidden":64,"attention":"after_each","fusion":"mean","loss":"mae_rmse"},
      {"branches":[["cnn","cnn"],["lstm","lstm"]],"hidden":64,"attention":"fusion","fusion":"gated","loss":"horizon_weighted_mae"},
      {"branches":[["transformer","transformer"]],"hidden":96,"attention":"input","fusion":"mean","loss":"huber"},
    ]
    return [validate_nas_spec({"model_name":"NAS_Composed","config":x}) for x in base]

def mutate_architecture(spec: dict[str, Any], *, seed: int, max_blocks: int = 5) -> dict[str, Any]:
    rng=random.Random(seed); c=deepcopy(validate_nas_spec(spec)["config"])
    op=rng.choice(("add_layer","remove_layer","swap_block","add_branch","remove_branch",
                   "attention","fusion","loss","width","optimizer","scheduler","learning_rate","weight_decay","dropout","loss_weight","meta_operator","mutate_meta_operator","darts_operator","mutate_darts_operator"))
    if op=="add_layer":
        b=rng.randrange(len(c["branches"]))
        if len(c["branches"][b]) < max_blocks:
            c["branches"][b].insert(rng.randrange(len(c["branches"][b])+1),rng.choice(BLOCKS))
    elif op=="remove_layer":
        options=[i for i,b in enumerate(c["branches"]) if len(b)>1]
        if options:
            b=rng.choice(options); c["branches"][b].pop(rng.randrange(len(c["branches"][b])))
    elif op=="swap_block":
        b=rng.randrange(len(c["branches"])); i=rng.randrange(len(c["branches"][b]))
        c["branches"][b][i]=rng.choice(BLOCKS)
    elif op=="add_branch" and len(c["branches"])<3:
        c["branches"].append([rng.choice(BLOCKS),rng.choice(BLOCKS)])
    elif op=="remove_branch" and len(c["branches"])>1:
        c["branches"].pop(rng.randrange(len(c["branches"])))
    elif op=="meta_operator":
        b=rng.randrange(len(c["branches"]))
        graph={"type":"meta_operator","ops":[rng.choice(META_PRIMITIVES) for _ in range(rng.randint(2,4))]}
        if len(c["branches"][b]) < max_blocks: c["branches"][b].insert(rng.randrange(len(c["branches"][b])+1),graph)
        else: c["branches"][b][rng.randrange(len(c["branches"][b]))]=graph
    elif op=="mutate_meta_operator":
        locations=[(bi,li) for bi,branch in enumerate(c["branches"]) for li,block in enumerate(branch)
                   if isinstance(block,dict) and block.get("type")=="meta_operator"]
        if locations:
            bi,li=rng.choice(locations)
            c["branches"][bi][li]=mutate_meta_operator_graph(c["branches"][bi][li],seed=rng.randrange(2**31))
        else:
            bi=rng.randrange(len(c["branches"]))
            if len(c["branches"][bi])<max_blocks:
                c["branches"][bi].append(propose_meta_operator_graph(weakness="general",seed=rng.randrange(2**31)))
    elif op=="math_operator":
        bi=rng.randrange(len(c["branches"]))
        graph=propose_math_operator_graph(weakness="general",seed=rng.randrange(2**31))
        if len(c["branches"][bi])<max_blocks: c["branches"][bi].insert(rng.randrange(len(c["branches"][bi])+1),graph)
        else: c["branches"][bi][rng.randrange(len(c["branches"][bi]))]=graph
    elif op=="mutate_math_operator":
        locations=[(bi,li) for bi,branch in enumerate(c["branches"]) for li,block in enumerate(branch)
                   if isinstance(block,dict) and block.get("type")=="math_operator"]
        if locations:
            bi,li=rng.choice(locations)
            c["branches"][bi][li]=mutate_math_operator_graph(c["branches"][bi][li],seed=rng.randrange(2**31))
        else:
            bi=rng.randrange(len(c["branches"]))
            if len(c["branches"][bi])<max_blocks:
                c["branches"][bi].append(propose_math_operator_graph(weakness="general",seed=rng.randrange(2**31)))
    elif op=="darts_operator":
        bi=rng.randrange(len(c["branches"]))
        graph=propose_darts_operator(seed=rng.randrange(2**31))
        if len(c["branches"][bi])<max_blocks: c["branches"][bi].insert(rng.randrange(len(c["branches"][bi])+1),graph)
        else: c["branches"][bi][rng.randrange(len(c["branches"][bi]))]=graph
    elif op=="mutate_darts_operator":
        locations=[(bi,li) for bi,branch in enumerate(c["branches"]) for li,block in enumerate(branch)
                   if isinstance(block,dict) and block.get("type")=="darts_operator"]
        if locations:
            bi,li=rng.choice(locations); old=c["branches"][bi][li]
            pool=[name for name in META_PRIMITIVES if name not in old["ops"]]
            changed=list(old["ops"])
            if pool: changed[rng.randrange(len(changed))]=rng.choice(pool)
            c["branches"][bi][li]={"type":"darts_operator","ops":changed}
        else:
            bi=rng.randrange(len(c["branches"]))
            if len(c["branches"][bi])<max_blocks: c["branches"][bi].append(propose_darts_operator(seed=rng.randrange(2**31)))
    elif op=="attention":
        c["attention"]=rng.choice(ATTENTION)
    elif op=="fusion":
        c["fusion"]=rng.choice(FUSION)
    elif op=="loss":
        c["loss"]=rng.choice(LOSSES)
    elif op=="width":
        c["hidden"]=rng.choice((32,48,64,96,128))
        c["attention_heads"]=next(h for h in (8,4,2,1) if c["hidden"]%h==0)
    elif op=="optimizer": c["optimizer"]=rng.choice(OPTIMIZERS)
    elif op=="scheduler": c["scheduler"]=rng.choice(SCHEDULERS)
    elif op=="learning_rate": c["learning_rate"]=rng.choice((1e-4,3e-4,5e-4,1e-3,2e-3,5e-3))
    elif op=="weight_decay": c["weight_decay"]=rng.choice((0.0,1e-5,1e-4,1e-3,1e-2))
    elif op=="dropout": c["dropout"]=rng.choice((0.0,0.05,0.1,0.2,0.3))
    elif op=="loss_weight": c["loss_alpha"]=rng.choice((0.1,0.25,0.5,0.75,1.0))
    try:
        return validate_nas_spec({"model_name":"NAS_Composed","config":c})
    except ValueError:
        return validate_nas_spec(spec)

def crossover_architectures(parent_a: dict[str, Any], parent_b: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Recombine independently validated branches and training recipes."""
    rng = random.Random(seed)
    a = validate_nas_spec(parent_a)["config"]; b = validate_nas_spec(parent_b)["config"]
    branches = deepcopy(rng.choice((a["branches"], b["branches"])))
    other = rng.choice((a, b))
    # Exchange one branch where possible; never exceed the registered topology budget.
    if rng.random() < .7 and other["branches"]:
        candidate = deepcopy(rng.choice(other["branches"]))
        if len(branches) < 3: branches.append(candidate)
        else: branches[rng.randrange(len(branches))] = candidate
    cfg = {
        "branches": branches[:3],
        "hidden": rng.choice((a["hidden"], b["hidden"])),
        "attention": rng.choice((a["attention"], b["attention"])),
        "attention_heads": 4,
        "fusion": rng.choice((a["fusion"], b["fusion"])),
        "dropout": rng.choice((a["dropout"], b["dropout"])),
        "loss": rng.choice((a["loss"], b["loss"])),
        "optimizer": rng.choice((a["optimizer"], b["optimizer"])),
        "learning_rate": rng.choice((a["learning_rate"], b["learning_rate"])),
        "weight_decay": rng.choice((a["weight_decay"], b["weight_decay"])),
        "scheduler": rng.choice((a["scheduler"], b["scheduler"])),
        "loss_alpha": rng.choice((a["loss_alpha"], b["loss_alpha"])),
    }
    cfg["attention_heads"] = next(h for h in (8, 4, 2, 1) if cfg["hidden"] % h == 0)
    return validate_nas_spec({"model_name": "NAS_Composed", "config": cfg})


def infer_structure_hypotheses(parent_spec: dict[str, Any], ablation_records: list[dict[str, Any]],
                                *, metric: str = "mae") -> list[dict[str, Any]]:
    """Turn paired ablation outcomes into explicit, falsifiable next-search hypotheses."""
    parent = validate_nas_spec(parent_spec)
    parent_id = _fingerprint(parent)
    parent_rows = [r for r in ablation_records if r.get("parent_architecture_id") == parent_id
                   and r.get("status") in ("completed", "success")
                   and isinstance(r.get("metrics", {}).get(metric), (int, float))]
    if not parent_rows: return []
    base = [float(r["parent_metric"]) for r in parent_rows if isinstance(r.get("parent_metric"), (int, float))]
    if not base: return []
    baseline = sum(base) / len(base)
    grouped = {}
    for row in parent_rows:
        key = json.dumps(row["config"], sort_keys=True, default=str)
        grouped.setdefault(key, []).append(float(row["metrics"][metric]))
    hypotheses = []
    for key, values in grouped.items():
        ablated = sum(values) / len(values); delta = ablated - baseline
        if abs(delta) < 1e-12: continue
        hypotheses.append({
            "statement": ("Removing this component improves validation performance; test leaner variants."
                          if delta < 0 else "This component appears useful; preserve it in offspring and test its placement."),
            "component_removed_spec": json.loads(key), "baseline_metric": baseline,
            "ablated_metric": ablated, "delta_ablation_minus_parent": delta,
            "evidence_strength": "preliminary_single_parent_ablation",
            "next_search_action": "favor_simplification" if delta < 0 else "preserve_component",
        })
    return sorted(hypotheses, key=lambda h: h["delta_ablation_minus_parent"])


def generate_nas_population(*, parents: list[dict[str, Any]] | None = None,
                            population_size: int = 12, seed: int = 2026,
                            max_params: int = 5_000_000) -> list[dict[str, Any]]:
    if not 1 <= population_size <= 100:
        raise ValueError("population_size must be in [1,100]")
    rng=random.Random(seed)
    raw_pool = list(parents or [])
    pool=[]
    for parent in raw_pool:
        try: pool.append(validate_nas_spec(parent, max_params=max_params))
        except (TypeError, ValueError): continue
    if not pool: pool=seed_architecture_population(seed)
    candidates=[]; seen=set()
    attempts=0
    while len(candidates)<population_size and attempts<population_size*30:
        attempts+=1
        parent=rng.choice(pool)
        if len(pool) > 1 and rng.random() < 0.25:
            mate = rng.choice([p for p in pool if p is not parent] or pool)
            child = crossover_architectures(parent, mate, seed=rng.randrange(2**31))
        else:
            child=mutate_architecture(parent,seed=rng.randrange(2**31))
        try: child=validate_nas_spec(child,max_params=max_params)
        except ValueError: continue
        fp=_fingerprint(child)
        if fp not in seen:
            seen.add(fp); child["architecture_id"]=fp
            candidates.append(child)
    return candidates

def propose_meta_operator_graph(*, weakness: str, seed: int = 2026, max_ops: int = 5) -> dict[str, Any]:
    """Create a bounded, declarative operator graph from a falsifiable error hypothesis."""
    rng=random.Random(seed); text=str(weakness).lower()
    if any(k in text for k in ("period","frequency","spectral","phase","周期","频域")):
        preferred=["spectral_gain","temporal_attention","layer_norm"]
    elif any(k in text for k in ("trend","drift","趋势","漂移")):
        preferred=["trend_split","depthwise_conv","gated_mlp","layer_norm"]
    elif any(k in text for k in ("variable","channel","correlation","变量","通道")):
        preferred=["channel_gate","gated_mlp","temporal_attention","layer_norm"]
    else:
        preferred=["depthwise_conv","gated_mlp","temporal_attention","layer_norm"]
    ops=list(preferred[:max_ops])
    if len(ops)<max_ops and rng.random()<0.5: ops.insert(rng.randrange(len(ops)+1),rng.choice(META_PRIMITIVES))
    return {"type":"meta_operator","ops":ops[:6]}

def mutate_meta_operator_graph(graph: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Edit a meta-operator graph using only registered primitive grammar rules."""
    rng=random.Random(seed)
    if not isinstance(graph,dict) or graph.get("type")!="meta_operator": raise ValueError("not a meta_operator graph")
    ops=list(graph.get("ops",[])); action=rng.choice(("insert","delete","replace","reorder"))
    if action=="insert" and len(ops)<6: ops.insert(rng.randrange(len(ops)+1),rng.choice(META_PRIMITIVES))
    elif action=="delete" and len(ops)>1: ops.pop(rng.randrange(len(ops)))
    elif action=="replace" and ops: ops[rng.randrange(len(ops))]=rng.choice(META_PRIMITIVES)
    elif action=="reorder" and len(ops)>1:
        i,j=rng.sample(range(len(ops)),2); ops[i],ops[j]=ops[j],ops[i]
    return {"type":"meta_operator","ops":ops}


def propose_math_operator_graph(*, weakness: str, seed: int = 2027) -> dict[str, Any]:
    """Discover a bounded mathematical operator from a falsifiable error hypothesis.

    The returned graph describes a registered equation template plus two primitive
    paths. It is data only; the model compiler implements the equation explicitly.
    """
    rng = random.Random(seed)
    text = str(weakness).lower()
    if any(k in text for k in ("period", "frequency", "spectral", "phase", "周期", "频域")):
        equation, ops, gate = "trend_spectral_mix", ["spectral_gain", "temporal_attention"], ["trend_split"]
    elif any(k in text for k in ("trend", "drift", "slow", "趋势", "漂移")):
        equation, ops, gate = "parallel_residual", ["depthwise_conv", "pointwise_mlp"], ["trend_split", "layer_norm"]
    elif any(k in text for k in ("variable", "channel", "correlation", "变量", "通道")):
        equation, ops, gate = "gated_residual", ["temporal_attention", "pointwise_mlp"], ["channel_gate"]
    else:
        equation, ops, gate = rng.choice(MATH_EQUATIONS), ["depthwise_conv", "pointwise_mlp"], ["gated_mlp"]
    # Small seeded variation makes repeated hypotheses explore nearby formulas.
    if rng.random() < 0.35:
        ops.insert(rng.randrange(len(ops)+1), rng.choice(META_PRIMITIVES))
    if equation == "residual_add": gate = []
    if equation == "trend_spectral_mix": gate = ["trend_split"]
    return {"type": "math_operator", "equation": equation,
            "ops": ops[:6], "gate_ops": gate[:4]}


def mutate_math_operator_graph(graph: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Mutate equation family or either computational path under the V7 grammar."""
    if not isinstance(graph, dict) or graph.get("type") != "math_operator":
        raise ValueError("not a math_operator graph")
    rng = random.Random(seed)
    result = {"type": "math_operator", "equation": graph.get("equation"),
              "ops": list(graph.get("ops", [])), "gate_ops": list(graph.get("gate_ops", []))}
    action = rng.choice(("equation", "main_insert", "main_replace", "main_delete", "gate_insert", "gate_replace"))
    if action == "equation":
        result["equation"] = rng.choice(MATH_EQUATIONS)
    elif action == "main_insert" and len(result["ops"]) < 6:
        result["ops"].insert(rng.randrange(len(result["ops"])+1), rng.choice(META_PRIMITIVES))
    elif action == "main_replace" and result["ops"]:
        result["ops"][rng.randrange(len(result["ops"]))] = rng.choice(META_PRIMITIVES)
    elif action == "main_delete" and len(result["ops"]) > 1:
        result["ops"].pop(rng.randrange(len(result["ops"])))
    elif action == "gate_insert" and len(result["gate_ops"]) < 4:
        result["gate_ops"].insert(rng.randrange(len(result["gate_ops"])+1), rng.choice(META_PRIMITIVES))
    elif action == "gate_replace" and result["gate_ops"]:
        result["gate_ops"][rng.randrange(len(result["gate_ops"]))] = rng.choice(META_PRIMITIVES)
    if result["equation"] in ("gated_residual", "multiplicative_modulation", "parallel_residual") and not result["gate_ops"]:
        result["gate_ops"] = ["gated_mlp"]
    if result["equation"] == "trend_spectral_mix":
        result["gate_ops"] = ["trend_split"]
    return result


def propose_operator_innovations(*, weakness: str, seed: int = 2026,
                                  parent_spec: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Propose safe, falsifiable operator variants from observed error patterns."""
    text = str(weakness).lower()
    parent = validate_nas_spec(parent_spec or seed_architecture_population(seed)[0])
    base = deepcopy(parent["config"])
    targets = []
    if any(k in text for k in ("period", "periodic", "frequency", "spectral", "phase", "周期", "频域")):
        targets.append(("amplitude_phase", "Test learnable spectral amplitude/phase correction for periodic or phase-shift errors."))
    if any(k in text for k in ("trend", "drift", "slow", "趋势", "漂移")):
        targets.append(("trend_residual", "Test explicit trend-residual decomposition for slow drift."))
    if any(k in text for k in ("cross-variable", "channel", "correlation", "变量", "通道")):
        targets.append(("cross_var", "Test cross-variable interaction for correlated sensor channels."))
    if not targets:
        targets = [("amplitude_phase", "Test whether learnable spectral amplitude/phase improves the incumbent."),
                   ("trend_residual", "Test whether trend-residual decomposition improves the incumbent.")]
    proposals=[]
    for block, rationale in targets:
        cfg=deepcopy(base)
        # Add the proposed operator to a new branch so its contribution can be ablated.
        if len(cfg["branches"]) < 3:
            cfg["branches"].append([block])
        else:
            cfg["branches"][-1] = [block]
        spec=validate_nas_spec({"model_name":"NAS_Composed","config":cfg})
        spec["proposal"]={"operator":block,"hypothesis":rationale,"source":"error_diagnosis"}
        proposals.append(spec)
    # V6: propose a genuinely new ordered primitive graph, not only a registered block.
    graph=propose_meta_operator_graph(weakness=weakness,seed=seed)
    graph_cfg=deepcopy(base)
    branch_index=min(len(graph_cfg["branches"])-1,2)
    if len(graph_cfg["branches"][branch_index]) < 5:
        graph_cfg["branches"][branch_index].append(graph)
    else:
        graph_cfg["branches"][branch_index][-1]=graph
    graph_spec=validate_nas_spec({"model_name":"NAS_Composed","config":graph_cfg})
    graph_spec["proposal"]={"operator":"meta_operator","graph":graph["ops"],
        "hypothesis":"Test an automatically composed primitive graph targeted to the diagnosed error.",
        "source":"bounded_meta_operator_grammar"}
    proposals.append(graph_spec)
    # V7: propose a new equation family with independently searchable main/gate paths.
    math_graph = propose_math_operator_graph(weakness=weakness, seed=seed + 101)
    math_cfg = deepcopy(base)
    math_branch = min(len(math_cfg["branches"]) - 1, 2)
    if len(math_cfg["branches"][math_branch]) < 5:
        math_cfg["branches"][math_branch].append(math_graph)
    else:
        math_cfg["branches"][math_branch][-1] = math_graph
    math_spec = validate_nas_spec({"model_name": "NAS_Composed", "config": math_cfg})
    math_spec["proposal"] = {"operator": "math_operator", "equation": math_graph["equation"],
        "main_path": math_graph["ops"], "gate_path": math_graph["gate_ops"],
        "hypothesis": "Test a newly composed mathematical interaction equation against the same primitive paths with ablation.",
        "source": "bounded_mathematical_operator_discovery_v7"}
    proposals.append(math_spec)
    # V8: differentiable mixed-operation edge; architecture logits receive gradients.
    darts_graph = propose_darts_operator(seed=seed + 211)
    darts_cfg = deepcopy(base)
    darts_branch = min(len(darts_cfg["branches"]) - 1, 2)
    if len(darts_cfg["branches"][darts_branch]) < 5:
        darts_cfg["branches"][darts_branch].append(darts_graph)
    else:
        darts_cfg["branches"][darts_branch][-1] = darts_graph
    darts_spec = validate_nas_spec({"model_name": "NAS_Composed", "config": darts_cfg})
    darts_spec["proposal"] = {"operator": "darts_operator", "candidate_ops": darts_graph["ops"],
        "hypothesis": "Train a differentiable softmax mixture of registered operators and discretize by learned architecture logits.",
        "source": "differentiable_mixed_operation_search_v8"}
    proposals.append(darts_spec)
    # Keep the three research-level innovations inside the default small population.
    proposals.sort(key=lambda item: {"darts_operator": 0, "math_operator": 1, "meta_operator": 2}.get(
        item.get("proposal", {}).get("operator"), 3))
    return proposals



def propose_darts_operator(*, seed: int = 2028, ops: list[str] | None = None) -> dict[str, Any]:
    """Create a bounded DARTS mixed-operation edge from registered primitives."""
    rng = random.Random(seed)
    pool = list(ops or META_PRIMITIVES)
    if len(pool) < 2 or any(op not in META_PRIMITIVES for op in pool):
        raise ValueError("DARTS candidate pool needs at least two registered primitives")
    selected = rng.sample(list(dict.fromkeys(pool)), k=min(DARTS_PRIMITIVE_LIMIT, len(set(pool))))
    return {"type": "darts_operator", "ops": selected}


def extract_darts_genotype(model, spec: dict[str, Any]) -> dict[str, Any]:
    """Export the argmax primitive from each trained DARTS edge as a discrete spec."""
    result = deepcopy(validate_nas_spec(spec))
    for bi, branch in enumerate(result["config"]["branches"]):
        for li, block in enumerate(branch):
            if isinstance(block, dict) and block.get("type") == "darts_operator":
                module = model.branches[bi][li]
                if not hasattr(module, "architecture_logits"):
                    raise ValueError("model/spec mismatch: missing DARTS architecture logits")
                index = int(module.architecture_logits.detach().argmax().cpu())
                result["config"]["branches"][bi][li] = {"type": "meta_operator", "ops": [block["ops"][index]]}
    return validate_nas_spec(result)


def search_symbolic_residual(inputs, residual, *, max_terms: int = 5, ridge: float = 1e-6):
    """Fit a small, interpretable symbolic residual model on a chronological holdout.

    The grammar is fixed and safe: raw variables, squares, absolute values,
    pairwise products, first differences, and low-order time/Fourier bases.
    Greedy forward selection is performed on the training prefix only; model
    selection uses the final chronological validation suffix.
    """
    from .symbolic_regression import search_symbolic_residual as _search
    return _search(inputs, residual, max_terms=max_terms, ridge=ridge)

def pareto_front(records: list[dict[str, Any]], *, objectives: tuple[str, ...] =
                 ("mae", "rmse", "parameter_count", "train_seconds")) -> list[dict[str, Any]]:
    """Return non-dominated architecture records; all available objectives minimize."""
    valid=[]
    for row in records:
        if row.get("status") not in ("completed", "success"):
            continue
        values={}
        for key in objectives:
            value=row.get("metrics",{}).get(key, row.get(key))
            if isinstance(value,(int,float)) and value >= 0:
                values[key]=float(value)
        if "mae" in values:
            valid.append((row,values))
    front=[]
    for i,(candidate,values) in enumerate(valid):
        dominated=False
        for j,(_,other) in enumerate(valid):
            if i==j: continue
            shared=set(values).intersection(other)
            if shared and all(other[k] <= values[k] for k in shared) and any(other[k] < values[k] for k in shared):
                dominated=True; break
        if not dominated: front.append(candidate)
    return front


def build_loss(name: str, *, horizon_weights=None, huber_delta: float = 1.0, loss_alpha: float = 0.25):
    """Return a torch loss callable; selected loss is recorded with every run."""
    import torch
    import torch.nn.functional as F
    if name not in LOSSES: raise ValueError(f"unsupported loss: {name}")
    def loss(pred, target):
        err=pred-target
        if name=="mae": return err.abs().mean()
        if name=="huber": return F.huber_loss(pred,target,delta=huber_delta)
        if name=="mae_rmse": return err.abs().mean()+loss_alpha*torch.sqrt(torch.mean(err**2)+1e-12)
        if name=="mae_slope":
            if pred.shape[-1] < 2: return err.abs().mean()
            slope_err=(pred[...,1:]-pred[...,:-1])-(target[...,1:]-target[...,:-1])
            return err.abs().mean()+loss_alpha*slope_err.abs().mean()
        weights=torch.as_tensor(horizon_weights,device=pred.device,dtype=pred.dtype) if horizon_weights is not None else torch.linspace(1,2,pred.shape[-1],device=pred.device,dtype=pred.dtype)
        if weights.numel()!=pred.shape[-1]: raise ValueError("horizon_weights length must equal prediction horizon")
        return (err.abs()*weights).mean()
    return loss

def build_optimizer(model, config: dict[str, Any]):
    """Instantiate the searched optimizer recipe for a PyTorch model."""
    import torch
    cfg = validate_nas_spec({"model_name": "NAS_Composed", "config": config})["config"]
    common = {"lr": cfg["learning_rate"], "weight_decay": cfg["weight_decay"]}
    if cfg["optimizer"] == "adamw": return torch.optim.AdamW(model.parameters(), **common)
    if cfg["optimizer"] == "adam": return torch.optim.Adam(model.parameters(), **common)
    return torch.optim.RMSprop(model.parameters(), **common)


def build_scheduler(optimizer, config: dict[str, Any]):
    """Instantiate the searched scheduler recipe; caller steps it per epoch."""
    import torch
    name = config.get("scheduler", "none")
    if name == "none": return None
    if name == "cosine": return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    if name == "plateau": return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)
    raise ValueError(f"unsupported scheduler: {name}")


def make_ablation_specs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate leave-one-component-out ablations for branches/attention/fusion."""
    base=validate_nas_spec(spec); c=deepcopy(base["config"]); out=[]
    for bi,branch in enumerate(c["branches"]):
        for li in range(len(branch)):
            if len(branch)>1:
                x=deepcopy(c); x["branches"][bi].pop(li)
                out.append(validate_nas_spec({"model_name":"NAS_Composed","config":x}))
    if c["attention"]!="none":
        x=deepcopy(c); x["attention"]="none"
        out.append(validate_nas_spec({"model_name":"NAS_Composed","config":x}))
    if len(c["branches"])>1:
        for fusion in FUSION:
            if fusion!=c["fusion"]:
                x=deepcopy(c); x["fusion"]=fusion
                out.append(validate_nas_spec({"model_name":"NAS_Composed","config":x}))
    # V7: ablate the discovered equation and its internal mathematical paths.
    for bi, branch in enumerate(c["branches"]):
        for li, block in enumerate(branch):
            if isinstance(block, dict) and block.get("type") == "darts_operator":
                for op_index in range(len(block["ops"])):
                    if len(block["ops"]) > 2:
                        x=deepcopy(c); x["branches"][bi][li]["ops"].pop(op_index)
                        out.append(validate_nas_spec({"model_name":"NAS_Composed","config":x}))
            if isinstance(block, dict) and block.get("type") == "math_operator":
                for path_name in ("ops", "gate_ops"):
                    path = block[path_name]
                    if len(path) > 1:
                        x=deepcopy(c); x["branches"][bi][li][path_name].pop()
                        out.append(validate_nas_spec({"model_name":"NAS_Composed","config":x}))
                # Equation-only control: retain the main path but remove the gate interaction.
                x=deepcopy(c); x["branches"][bi][li]["equation"]="residual_add"
                x["branches"][bi][li]["gate_ops"]=[]
                out.append(validate_nas_spec({"model_name":"NAS_Composed","config":x}))
    # One-factor training-recipe controls distinguish structural gains from optimizer/loss effects.
    for loss_name in LOSSES:
        if loss_name != c["loss"]:
            x=deepcopy(c); x["loss"]=loss_name
            out.append(validate_nas_spec({"model_name":"NAS_Composed","config":x}))
    for optimizer_name in OPTIMIZERS:
        if optimizer_name != c["optimizer"]:
            x=deepcopy(c); x["optimizer"]=optimizer_name
            out.append(validate_nas_spec({"model_name":"NAS_Composed","config":x}))
    if not out:
        # If the parent is already minimal, test a controlled block substitution.
        x=deepcopy(c); x["branches"][0][0]="linear" if x["branches"][0][0]!="linear" else "cnn"
        out.append(validate_nas_spec({"model_name":"NAS_Composed","config":x}))
    unique={}
    for x in out: unique[_fingerprint(x)]=x
    return list(unique.values())

def rank_experiments(records: list[dict[str, Any]], metric: str="mae") -> list[dict[str, Any]]:
    valid=[r for r in records if r.get("status") in ("completed","success")
           and isinstance(r.get("metrics",{}).get(metric),(int,float))]
    return sorted(valid,key=lambda r:float(r["metrics"][metric]))
