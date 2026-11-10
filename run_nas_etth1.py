"""Run real-data ETTh1 neural architecture search (validation only)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from atrea.nas_training import run_etth1_nas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--device", default=None, help="cpu, cuda, or auto (default: auto)")
    parser.add_argument("--weakness", default="periodic frequency phase drift", help="error diagnosis used to propose novel operators")
    parser.add_argument("--output", default="output/nas/etth1_nas_results.json")
    args = parser.parse_args()
    result = run_etth1_nas(population_size=args.population_size,
        generations=args.generations, epochs=args.epochs, batch_size=args.batch_size,
        patience=args.patience, seeds=args.seeds,
        device=None if args.device in (None, "auto") else args.device,
        weakness=args.weakness)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(path), "split_hash": result["split_hash"],
        "experiment_count": len(result["experiments"]),
        "ablation_count": result["ablation_count"],
        "best_candidate": result["best_candidate"],
        "promotion_allowed": result["promotion_allowed"]}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
