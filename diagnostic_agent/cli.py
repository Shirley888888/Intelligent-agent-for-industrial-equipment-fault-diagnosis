from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dual_evolution.episode_store import EpisodeStore
from dual_evolution.knowledge_catalog import KnowledgeCatalog
from dual_evolution.runtime_guard import RuntimeGuardConfig

from .agent import AgentRuntimeConfig, DiagnosticAgent
from .providers import OfflinePlanner, OpenAICompatiblePlanner
from .model_bridge import VersionedForecastModelBridge
from .industrial_fault import IndustrialFaultTrainConfig, train_industrial_fault_model
from .model_router import TaskAwareModelRouter
from .ett_corpus_bridge import ETTOilTemperatureBridge
from .chronos2_oil_bridge import Chronos2OilTemperatureBridge


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _bootstrap_demo_knowledge(catalog: KnowledgeCatalog) -> None:
    """Add a conspicuously synthetic doc to exercise the approved-RAG path only."""
    existing = {(x["document_id"], x["version"]) for x in catalog._documents().values()}
    key = ("SYNTHETIC-DEMO-PUMP", "0.0-demo")
    if key in existing:
        return
    catalog.register_document({
        "document_id": key[0], "version": key[1],
        "title": "合成演示资料（非真实设备手册）",
        "source_uri": "demo://synthetic-not-a-real-manual",
        "equipment_scope": ["PUMP-DEMO-001"], "effective_from": "2026-01-01",
        "effective_to": None, "summary": "仅用于验证知识检索和引用链路，不包含可执行维修步骤。",
        "content": "合成演示资料：泵设备 轴承 温度 振动 巡检。该资料不代表任何厂家规范，不得用于现场维修。",
        "keywords": ["泵", "轴承", "温度", "振动"], "registered_by": "demo-fixture",
    })
    catalog.approve_document(key[0], key[1], reviewer="demo-fixture",
                             approval_ref="demo://synthetic-approval-not-for-production")


def build_agent(workspace: Path, provider: str = "offline") -> DiagnosticAgent:
    workspace.mkdir(parents=True, exist_ok=True)
    knowledge = KnowledgeCatalog(workspace / "knowledge_catalog.jsonl")
    episode_store = EpisodeStore(workspace / "episodes.jsonl")
    planner = OpenAICompatiblePlanner.from_env() if provider == "openai" else OfflinePlanner()
    # The sample calibration ID is a demo placeholder. Production must inject a validated task/equipment calibration.
    guard = RuntimeGuardConfig(ood_warn=0.70, ood_critical=0.90,
                               drift_warn=0.20, drift_critical=0.30,
                               min_calibrated_confidence=0.70,
                               calibration_id=os.getenv("AGENT_CALIBRATION_ID", "UNSET"))
    config = AgentRuntimeConfig(runtime_guard=guard,
                                strategy_min_support=int(os.getenv("AGENT_STRATEGY_MIN_SUPPORT", "2")))
    return DiagnosticAgent(planner=planner, knowledge=knowledge,
                           episode_store=episode_store, workspace=workspace, config=config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Industrial equipment diagnostic Agent")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run a synthetic offline end-to-end example")
    demo.add_argument("--workspace", default="agent_workspace_demo")
    run = sub.add_parser("run", help="Run diagnosis from a structured model prediction JSON")
    run.add_argument("--input", required=True)
    run.add_argument("--workspace", default="agent_workspace")
    run.add_argument("--provider", choices=["offline", "openai"], default="offline")
    model_run = sub.add_parser("run-model", help="Run the active versioned LSTM/TCN forecast model then Agent")
    model_run.add_argument("--model-root", required=True, help="VersionManager directory with current.json and versions/")
    model_run.add_argument("--csv", default="ETTh1.csv", help="Observed 7-feature ETTh1-format CSV")
    model_run.add_argument("--equipment-id", default="ETTH1-REPLAY")
    model_run.add_argument("--operating-condition", default="dataset_replay")
    model_run.add_argument("--workspace", default="agent_workspace_model_bridge")
    model_run.add_argument("--device", default="cpu")
    model_run.add_argument("--provider", choices=["offline", "openai"], default="offline")
    feedback = sub.add_parser("feedback", help="Submit a reviewed result for an existing episode")
    feedback.add_argument("--workspace", default="agent_workspace")
    feedback.add_argument("--episode-id", required=True)
    feedback.add_argument("--reviewer", required=True)
    feedback.add_argument("--source-ref", required=True)
    feedback.add_argument("--status", choices=["verified", "disputed", "invalid"], required=True)
    feedback.add_argument("--ground-truth", help="JSON value; required for verified")
    feedback.add_argument("--corrected-decision")
    feedback.add_argument("--failure-categories", default="")
    feedback.add_argument("--note", default="")
    task_run = sub.add_parser("run-task", help="Route either ETTh1 forecast or industrial fault model into the same Agent")
    task_run.add_argument("--config", required=True, help="Task JSON: task_type + task-specific model and input paths")
    task_run.add_argument("--workspace", default="agent_workspace_unified")
    task_run.add_argument("--device", default="cpu")
    task_run.add_argument("--provider", choices=["offline", "openai"], default="offline")
    ett_run = sub.add_parser("run-ett-corpus", help="Run a trained multi-transformer ETT oil-temperature model through the shared Agent")
    ett_run.add_argument("--model-dir", required=True, help="Folder containing checkpoint.pt and manifest.json")
    ett_run.add_argument("--csv", required=True, help="ETTh1/ETTh2/ETTm1/ETTm2-format CSV")
    ett_run.add_argument("--equipment-id", required=True)
    ett_run.add_argument("--operating-condition", default="dataset_replay")
    ett_run.add_argument("--workspace", default="agent_workspace_ett")
    ett_run.add_argument("--device", default="cpu")
    ett_run.add_argument("--provider", choices=["offline", "openai"], default="offline")
    chronos_run = sub.add_parser("run-chronos2-oil", help="Run a fine-tuned Chronos-2 oil-temperature model through the shared Agent")
    chronos_run.add_argument("--model-dir", required=True, help="Chronos-2 fine-tune output with manifest.json and finetuned-ckpt/")
    chronos_run.add_argument("--csv", required=True, help="ETT-format oil-temperature history CSV")
    chronos_run.add_argument("--equipment-id", required=True)
    chronos_run.add_argument("--operating-condition", default="dataset_replay")
    chronos_run.add_argument("--workspace", default="agent_workspace_chronos2")
    chronos_run.add_argument("--device-map", default="auto")
    chronos_run.add_argument("--provider", choices=["offline", "openai"], default="offline")
    train_fault = sub.add_parser("train-industrial-model", help="Train a labelled industrial multi-task LSTM/TCN and register M0")
    train_fault.add_argument("--dataset", required=True, help="NPZ with separate X_train/X_val and labels")
    train_fault.add_argument("--config", required=True, help="Industrial fault training config JSON")
    train_fault.add_argument("--out", required=True, help="New model registry directory; must be empty or absent")
    candidates = sub.add_parser("strategy-candidates", help="List unactivated Prompt/tool/retrieval candidates")
    candidates.add_argument("--workspace", default="agent_workspace")
    args = parser.parse_args()
    workspace = Path(getattr(args, "workspace", "agent_workspace"))
    if args.command == "train-industrial-model":
        raw_cfg = _load_json(args.config)
        if "tcn_channels" in raw_cfg:
            raw_cfg["tcn_channels"] = tuple(raw_cfg["tcn_channels"])
        cfg = IndustrialFaultTrainConfig(**raw_cfg)
        result = train_industrial_fault_model(args.dataset, cfg, args.out)
    elif args.command == "feedback":
        agent = build_agent(workspace, "offline")
        gt = json.loads(args.ground_truth) if args.ground_truth else None
        result = agent.submit_feedback(args.episode_id, reviewer=args.reviewer,
            source_ref=args.source_ref, status=args.status, ground_truth=gt,
            corrected_decision=args.corrected_decision,
            failure_categories=[x.strip() for x in args.failure_categories.split(",") if x.strip()],
            note=args.note)
    elif args.command == "strategy-candidates":
        agent = build_agent(workspace, "offline")
        result = agent.strategy_candidates.list_candidates()
    elif args.command == "run-ett-corpus":
        agent = build_agent(workspace, args.provider)
        bridge = ETTOilTemperatureBridge(args.model_dir, device=args.device)
        result = bridge.run(agent, args.csv, equipment_id=args.equipment_id,
                            operating_condition=args.operating_condition)
    elif args.command == "run-chronos2-oil":
        agent = build_agent(workspace, args.provider)
        bridge = Chronos2OilTemperatureBridge(args.model_dir, device_map=args.device_map)
        result = bridge.run(agent, args.csv, equipment_id=args.equipment_id,
                            operating_condition=args.operating_condition)
    elif args.command == "run-model":
        agent = build_agent(workspace, getattr(args, "provider", "offline"))
        bridge = VersionedForecastModelBridge(args.model_root, device=args.device)
        result = bridge.run(agent, args.csv, equipment_id=args.equipment_id,
                            operating_condition=args.operating_condition)
    elif args.command == "run-task":
        agent = build_agent(workspace, args.provider)
        result = TaskAwareModelRouter(device=args.device).run(agent, _load_json(args.config))
    else:
        agent = build_agent(workspace, getattr(args, "provider", "offline"))
        if args.command == "demo":
            _bootstrap_demo_knowledge(agent.knowledge)
            context = {
                "episode_id": "DEMO-AGENT-0001", "prediction_time": "2026-11-02T08:00:00+08:00",
                "equipment_id": "PUMP-DEMO-001", "operating_condition": "steady_load",
                "sensor_readings": {"bearing_temp_c": [68.1, 69.0, 70.2],
                                    "vibration_mm_s": [2.1, 2.3, 2.2]},
                "model": {"model_name": "TCN", "model_version": "M0", "activation_id": 1},
                "model_prediction": {"fault_type": "bearing_fault_candidate", "anomaly_score": 0.81,
                    "confidence": 0.86, "confidence_calibrated": True, "ood_score": 0.12,
                    "drift_score": 0.08, "rul_hours": 36.0, "fault_amplitude": 0.42},
            }
        else:
            context = _load_json(args.input)
        result = agent.run(context)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
