#!/usr/bin/env python3
"""Run a local fine-tuned Chronos-2 oil-temperature forecaster through the shared Agent."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from diagnostic_agent.chronos2_oil_bridge import Chronos2OilTemperatureBridge
from diagnostic_agent.cli import build_agent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="Chronos2 frequency output dir, e.g. models/chronos2_ett/hourly")
    p.add_argument("--csv", required=True, help="ETT-format recent history CSV")
    p.add_argument("--equipment-id", required=True)
    p.add_argument("--workspace", default="agent_workspace_chronos2")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--provider", choices=["offline", "openai"], default="offline")
    p.add_argument("--operating-condition", default="dataset_replay")
    args = p.parse_args()
    agent = build_agent(Path(args.workspace), args.provider)
    bridge = Chronos2OilTemperatureBridge(args.model_dir, device_map=args.device_map)
    result = bridge.run(agent, args.csv, equipment_id=args.equipment_id,
                        operating_condition=args.operating_condition)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
