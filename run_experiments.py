"""Four-arm, controlled-concept-drift experiment driver for the ETTh1 online loop.

The raw ETTh1.csv is read-only. Drift is applied to a copy of the loaded ndarray
only from online_start + offset_episodes*24 onward, and is shared by all arms.
"""
from __future__ import annotations
import csv, json, math, os, shutil, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from minimal_loop import (GateConfig, TrainConfig, load_csv, simulate, write_json,
                          append_json, FEATURES, H, TARGET)

# modify:canonical-config - experiments/config.yaml is the single user-editable source of thresholds.
# EXPERIMENTS_CONFIG lets a second variant (e.g. GPU) reuse this driver without editing the default config.
CONFIG_PATH = Path(os.environ.get("EXPERIMENTS_CONFIG") or (HERE / "experiments" / "config.yaml"))
# JSON is a valid YAML subset; standard-library parsing avoids a PyYAML dependency.
CONFIG = json.loads("\n".join(line for line in CONFIG_PATH.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#")))


def drift_copy(values: np.ndarray, online_start: int, cfg: dict) -> tuple[np.ndarray, dict]:
    result = values.copy()
    if not cfg.get("enabled", False):
        return result, {"enabled": False}
    cutoff = online_start + int(cfg["offset_episodes"]) * H
    require = lambda ok, msg: (_ for _ in ()).throw(ValueError(msg)) if not ok else None
    require(0 < cutoff < len(result), "Drift cutoff must fall inside the data")
    ref_start = max(0, cutoff-int(cfg["reference_hours"]))
    reference_mean = float(np.mean(result[ref_start:cutoff, TARGET]))
    scale = float(cfg["variance_scale"])
    shift = float(cfg["mean_shift_c"])
    # modify:drift-injection - OT only, after cutoff; residual spread is expanded around pre-drift mean.
    result[cutoff:, TARGET] = reference_mean + shift + (result[cutoff:, TARGET] - reference_mean) * scale
    return result, {"enabled": True, "cutoff_idx": cutoff, "cutoff_episode": int(cfg["offset_episodes"]),
                    "reference_range": [ref_start, cutoff], "reference_mean_ot": reference_mean,
                    "mean_shift_c": shift, "variance_scale": scale,
                    "affected_channel": "OT", "original_csv_modified": False}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists(): return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def episode_metrics(run_dir: Path, baseline: float) -> dict:
    eps = [e for e in load_jsonl(run_dir/"feedback.jsonl") if e.get("status") == "complete"]
    errors = np.array([np.asarray(e["forecast"], dtype=float)-np.asarray(e["truth"], dtype=float) for e in eps])
    if len(errors):
        mae_ep = np.mean(np.abs(errors), axis=1)
        overall_mae = float(np.mean(np.abs(errors)))
        overall_mse = float(np.mean(errors**2))
        worst = float(max(np.mean(mae_ep[i:i+10]) for i in range(max(1, len(mae_ep)-9))))
    else:
        mae_ep = np.array([]); overall_mae = overall_mse = worst = float("nan")
    # Recovery: first 10-episode moving MAE > 1.2*baseline, then first later <= 1.1*baseline.
    rolling = np.array([np.mean(mae_ep[i-9:i+1]) for i in range(9, len(mae_ep))]) if len(mae_ep) >= 10 else np.array([])
    high = np.flatnonzero(rolling > baseline*1.2)
    recovery = None
    if len(high):
        first = int(high[0])
        lows = np.flatnonzero((np.arange(len(rolling)) > first) & (rolling <= baseline*1.1))
        if len(lows): recovery = int(lows[0]-first)
    summary = json.loads((run_dir/"summary.json").read_text(encoding="utf-8"))
    retrain_events = load_jsonl(run_dir/"retrain_events.jsonl")
    version_events = load_jsonl(run_dir/"version_events.jsonl")
    upgrades = [e for e in version_events if e.get("action") in ("upgrade", "bootstrap")]
    rollbacks = [e for e in version_events if e.get("action") == "rollback"]
    trace = load_jsonl(run_dir/"react_trace.jsonl")
    pred_n = max(1, len(load_jsonl(run_dir/"predictions.jsonl")))
    # Fixed-policy baselines are credited one scheduling decision per episode, per protocol.
    mean_steps = (1.0 if run_dir.name in ("frozen", "periodic") else (len(trace)/pred_n if trace else 0.0))
    return {"complete_episodes": len(eps), "overall_prequential_mae": overall_mae,
            "overall_prequential_mse": overall_mse, "worst_10_episode_mae": worst,
            "retrain_triggers": len(retrain_events), "actual_activations": max(0, len(upgrades)-1),
            "rollback_count": len(rollbacks), "recovery_episodes": recovery,
            "mean_decision_steps_per_episode": mean_steps,
            "baseline_mae": baseline, "rolling_mae": rolling.tolist(),
            "episode_mae": mae_ep.tolist(), "predictions": len(load_jsonl(run_dir/"predictions.jsonl")),
            "summary": summary, "retrain_events": retrain_events, "version_events": version_events,
            "trace": trace, "episodes": eps}


def save_tables(results: dict[str, dict], out: Path):
    fields = ["arm", "overall_prequential_mae", "overall_prequential_mse", "worst_10_episode_mae",
              "retrain_triggers", "actual_activations", "rollback_count", "recovery_episodes",
              "mean_decision_steps_per_episode", "complete_episodes"]
    with (out/"summary_table.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for name, r in results.items():
            w.writerow({"arm": name, **{k: r[k] for k in fields if k != "arm"}})
    lines = ["# Four-arm online learning experiment summary", "",
             "Controlled concept drift is injected only into the loaded ndarray after the configured cutoff; the source CSV is unchanged.", "",
             "| Arm | Overall MAE | Overall MSE | Worst 10-episode MAE | Retrain triggers | Activations | Rollbacks | Recovery (episodes) | Mean decision steps / episode | Complete episodes |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        rec = "N/A" if r["recovery_episodes"] is None else str(r["recovery_episodes"])
        lines.append(f"| {name} | {r['overall_prequential_mae']:.6f} | {r['overall_prequential_mse']:.6f} | {r['worst_10_episode_mae']:.6f} | {r['retrain_triggers']} | {r['actual_activations']} | {r['rollback_count']} | {rec} | {r['mean_decision_steps_per_episode']:.2f} | {r['complete_episodes']} |")
    lines += ["", "Recovery is measured from the first 10-episode rolling MAE above baseline×1.2 to the first later value at or below baseline×1.1. N/A means no qualifying recovery was observed before replay ended."]
    (out/"summary_table.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def save_figures(results: dict[str, dict], out: Path):
    # 1. MAE timeline
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, r in results.items():
        rolling = np.asarray(r["rolling_mae"])
        if len(rolling): ax.plot(np.arange(9, 9+len(rolling)), rolling, label=name, linewidth=1.8)
    ax.set_xlabel("Episode index"); ax.set_ylabel("10-episode rolling MAE (C)")
    ax.set_title("Prequential MAE under controlled concept drift"); ax.grid(True, alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(out/"fig_mae_timeline.png", dpi=160); plt.close(fig)
    # 2. Arm 3 version timeline
    r = results["Arm 3 - Feedback-Triggered ReAct"]
    fig, ax = plt.subplots(figsize=(11, 6))
    rolling = np.asarray(r["rolling_mae"])
    if len(rolling): ax.plot(np.arange(9, 9+len(rolling)), rolling, label="Agent rolling MAE", linewidth=2)
    for ev in r["version_events"]:
        action = ev.get("action", "")
        if action in ("upgrade", "rollback"):
            # Episode coordinate from ISO time is mapped to the event's activation timestamp against prediction times.
            pred_times = [e["prediction_time"] for e in r["episodes"]]
            x = next((i for i, ts in enumerate(pred_times) if ts >= ev.get("activated_as_of", "")), len(pred_times)-1)
            ax.axvline(x, linestyle="--", alpha=.7, label=f"{action}: {ev.get('model_version','?')}")
    ax.set_xlabel("Episode index"); ax.set_ylabel("10-episode rolling MAE (C)")
    ax.set_title("Arm 3: version activation events"); ax.grid(True, alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(out/"fig_version_timeline.png", dpi=160); plt.close(fig)
    # 3. grouped comparison bars, metrics use separate normalized panels only to preserve legibility.
    names = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric, title in zip(axes,
        ["overall_prequential_mae", "worst_10_episode_mae", "retrain_triggers"],
        ["Overall MAE", "Worst 10-episode MAE", "Retrain triggers"]):
        vals = [results[n][metric] for n in names]
        ax.bar(range(len(names)), vals); ax.set_xticks(range(len(names)), [f"Arm {i+1}" for i in range(len(names))], rotation=0)
        ax.set_title(title); ax.grid(axis="y", alpha=.25)
    fig.suptitle("Ablation comparison"); fig.tight_layout(); fig.savefig(out/"fig_ablation_bar.png", dpi=160); plt.close(fig)
    # 4. Representative ReAct trace decisions as a simple readable event timeline.
    trace = r["trace"]
    fig, ax = plt.subplots(figsize=(11, 4))
    labels = ["observe", "monitor", "train", "gate", "publish/reject", "predict"]
    ymap = {"observe_clock":0,"release_matured_feedback":1,"inspect_recent_errors":2,
            "maybe_train_candidate":3,"run_gate":4,"publish_or_reject":5,"issue_prediction":6}
    xs, ys = [], []
    for i, item in enumerate(trace):
        if item.get("tool_name") in ymap:
            xs.append(i); ys.append(ymap[item["tool_name"]])
    if xs: ax.scatter(xs, ys, s=8)
    ax.set_yticks(range(7), ["observe", "release feedback", "monitor", "train candidate", "gate", "publish/reject", "predict"])
    ax.set_xlabel("Trace record index"); ax.set_title("Arm 3 ReAct tool sequence"); ax.grid(True, axis="x", alpha=.2)
    fig.tight_layout(); fig.savefig(out/"fig_react_trace_sample.png", dpi=160); plt.close(fig)


def human_logs(results: dict[str, dict], out: Path):
    lines = ["# Model update log", "", "Times are ETTh1 dataset-local timestamps. Gate scores are the recent/historical incumbent and candidate MAEs when a gate was evaluated.", ""]
    for name, r in results.items():
        for e in r["version_events"]:
            if e.get("action") == "bootstrap": continue
            previous = (e.get("previous") or {}).get("model_version", "None")
            new = e.get("model_version", "?")
            baseline = e.get("baseline_mae", float("nan"))
            gate = "not run (ablation policy)"
            for te in r["retrain_events"]:
                if te.get("as_of") == e.get("activated_as_of") and te.get("gate"):
                    g=te["gate"]
                    gate=f"recent {g.get('incumbent_recent_mae')} → {g.get('candidate_recent_mae')}; historical {g.get('incumbent_historical_mae')} → {g.get('candidate_historical_mae')}"
            lines.append(f"- **{name}** — {e.get('activated_as_of')}: {previous} → {new}; action={e.get('action')}; gate={gate}; baseline MAE={baseline:.6f}.")
        for te in r["retrain_events"]:
            if te.get("new_version") is None:
                g=te.get("gate")
                detail = (f"recent_ok={g.get('recent_ok')}, historical_ok={g.get('historical_ok')}" if g else te.get("error", "candidate not published"))
                lines.append(f"- **{name}** — {te.get('as_of')}: candidate {te.get('candidate_id')} rejected/not activated; {detail}.")
    (out/"model_update_log.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def trace_examples(results: dict[str, dict], out: Path):
    trace=results["Arm 3 - Feedback-Triggered ReAct"]["trace"]
    wants=[("Normal prediction", lambda x:x.get("tool_name")=="issue_prediction"),
           ("Degradation detection", lambda x:x.get("tool_name")=="inspect_recent_errors" and x.get("tool_result",{}).get("should_attempt")),
           ("Gate rejection", lambda x:x.get("tool_name")=="run_gate" and x.get("tool_result",{}).get("allow_upgrade") is False),
           ("Gate pass / publish", lambda x:x.get("tool_name")=="run_gate" and x.get("tool_result",{}).get("allow_upgrade") is True)]
    lines=["# Representative ReAct trace examples", "", "These excerpts are selected from the actual Arm 3 JSONL trace; if a scenario did not occur, that is stated rather than fabricated.", ""]
    for label,pred in wants:
        found=next((x for x in trace if pred(x)),None)
        lines.append(f"## {label}\n")
        if found: lines += ["```json",json.dumps(found,ensure_ascii=False,indent=2),"```",""]
        else: lines += ["Not observed in this replay.",""]
    (out/"react_trace_examples.md").write_text("\n".join(lines),encoding="utf-8")


def conclusion(results: dict[str, dict], drift: dict, out: Path):
    sorted_arms=sorted(results.items(),key=lambda x:x[1]["overall_prequential_mae"])
    best=sorted_arms[0][0]
    a3=results["Arm 3 - Feedback-Triggered ReAct"]; a4=results["Arm 4 - Feedback-Triggered No-Gate"]
    a1=results["Arm 1 - Frozen Model"]; a2=results["Arm 2 - Periodic Retrain"]
    def fmt(x): return "未观察到恢复" if x is None else f"{x} 个 Episode"
    text=(f"本实验在 ETTh1 加载后的副本上实施受控概念漂移，起点为行 {drift.get('cutoff_idx')}，仅对 OT 通道施加均值偏移 {drift.get('mean_shift_c')}℃ 并将偏差幅度放大至 {drift.get('variance_scale')} 倍；原始 CSV 未修改。整体 prequential MAE 最低的是 {best}。ReAct 的 MAE 为 {a3['overall_prequential_mae']:.4f}，Frozen 为 {a1['overall_prequential_mae']:.4f}，Periodic 为 {a2['overall_prequential_mae']:.4f}，因此应依据数值而非机制预期判断反馈触发策略是否优于固定模型和定期重训。\n\nGate 对比方面，Arm 3 MAE 为 {a3['overall_prequential_mae']:.4f}，去 Gate 的 Arm 4 为 {a4['overall_prequential_mae']:.4f}；差值（No-Gate 减 ReAct）为 {a4['overall_prequential_mae']-a3['overall_prequential_mae']:.4f}。正值表示启用 Gate 的误差较低，负值则表示本次回放中 Gate 未带来 MAE 优势；还需结合最差阶段 MAE 与上线次数解释。恢复速度按 10-Episode 滑动 MAE 从 baseline×1.2 以上回到 baseline×1.1 以内定义：Arm 3 为 {fmt(a3['recovery_episodes'])}，Frozen 为 {fmt(a1['recovery_episodes'])}，Periodic 为 {fmt(a2['recovery_episodes'])}，No-Gate 为 {fmt(a4['recovery_episodes'])}。若未恢复或未出现退化，不将其伪装为零。该结论仅适用于本次受控漂移强度和固定切分，不能直接外推为 ETTh1 原始数据存在自然概念漂移。")
    # 300-500 Chinese characters is requested; keep this as a single compact, evidence-driven paragraph.
    (out/"conclusion.md").write_text("# 实验结论\n\n"+text+"\n",encoding="utf-8")


def main():
    # EXPERIMENTS_ROOT keeps variant runs in their own directory so the default records survive.
    out = HERE / (os.environ.get("EXPERIMENTS_ROOT") or CONFIG["output"]["root"])
    out = out.resolve()
    require_outside = HERE.resolve() / "experiments"
    if out == require_outside and str(CONFIG_PATH) != str(HERE / "experiments" / "config.yaml"):
        raise SystemExit("A non-default config must write to its own EXPERIMENTS_ROOT")
        # modify:preserve-config-and-entrypoints - reruns clear only generated results.
        keep = {"config.yaml", "README.md", "minimal_loop.py", "run_experiments.py"}
        for child in list(out.iterdir()):
            if child.name in keep: continue
            if child.is_dir(): shutil.rmtree(child)
            else: child.unlink()
    out.mkdir(parents=True, exist_ok=True)
    raw,times=load_csv(HERE/"ETTh1.csv")
    split=CONFIG["split"]
    values,drift=drift_copy(raw,split["online_start"],CONFIG["drift"])
    write_json(out/"drift_manifest.json",drift)
    train=TrainConfig(**CONFIG["train_config"]); gate=GateConfig(**CONFIG["gate"])
    mon=CONFIG["monitor"]
    arms=[
      ("Arm 1 - Frozen Model","frozen"),
      ("Arm 2 - Periodic Retrain","periodic"),
      ("Arm 3 - Feedback-Triggered ReAct","react"),
      ("Arm 4 - Feedback-Triggered No-Gate","no_gate"),
    ]
    results={}
    for name,strategy in arms:
        print(f"[run] {name}",flush=True)
        arm_dir=out/{"frozen":"frozen","periodic":"periodic","react":"agent","no_gate":"no_gate_ablation"}[strategy]
        summary=simulate(values,times,arm_dir,config=TrainConfig(**CONFIG["train_config"]),
            monitor_n=mon["n"],patience=mon["patience"],monitor_relative=mon["relative"],
            monitor_absolute=mon["absolute"],cooldown_episodes=mon["cooldown_episodes"],
            gate_config=GateConfig(**CONFIG["gate"]),strategy=strategy,periodic_k=CONFIG["periodic"]["k"],
            train_end=split["train_end"],online_start=split["online_start"],
            candidate_validation_hours=CONFIG["split"].get("candidate_validation_hours", 168))
        # Stable output contract: every arm has all expected log files, even when empty.
        for fn in ("predictions.jsonl","feedback.jsonl","monitor.jsonl","retrain_events.jsonl","version_events.jsonl","react_trace.jsonl"):
            (arm_dir/fn).touch(exist_ok=True)
        results[name]=episode_metrics(arm_dir,summary["baseline_mae"])
        print(f"  episodes={results[name]['complete_episodes']} MAE={results[name]['overall_prequential_mae']:.5f} triggers={results[name]['retrain_triggers']} activations={results[name]['actual_activations']}",flush=True)
    # Save compact serializable results separately from internal trace detail.
    save_tables(results,out)
    save_figures(results,out)
    human_logs(results,out); trace_examples(results,out); conclusion(results,drift,out)
    compact={name:{k:v for k,v in r.items() if k not in ("trace","episodes","summary","retrain_events","version_events")} for name,r in results.items()}
    write_json(out/"metrics_detail.json",compact)
    print(json.dumps({name:{k:r[k] for k in ("overall_prequential_mae","overall_prequential_mse","worst_10_episode_mae","retrain_triggers","actual_activations","rollback_count","recovery_episodes","mean_decision_steps_per_episode")} for name,r in results.items()},indent=2))

if __name__=="__main__": main()
