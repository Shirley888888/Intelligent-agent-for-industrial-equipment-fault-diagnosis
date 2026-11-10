"""modify:torch-smoke - actual train/retrain/gate/rollback check on a user's PyTorch environment.

This tiny ETTh1 prefix run checks execution only, not forecasting improvements.
No simulated performance scores are used. Output directory must be new.
--version-checks additionally exercises upgrade/rollback/stale-gate/corruption
paths; those gate scores are controlled test inputs, not model gains.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from minimal_loop import (GateConfig, TrainConfig, VersionManager, digest, load_csv,
                          model_gate, require, simulate, write_json)


def controlled_gate(manager: VersionManager, candidate: dict) -> dict:
    # Controlled scores bound to the actual current activation; test inputs only.
    current = manager.current()
    model, _ = manager.load(current["model_version"])
    return {**model_gate(1.0, 0.90, 1.0, 1.0),
            "incumbent_version": current["model_version"],
            "incumbent_activation_id": current["activation_id"],
            "incumbent_checkpoint_sha256": digest(model.checkpoint),
            "candidate_id": candidate["candidate_id"],
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "snapshot_sha256": candidate["manifest"]["snapshot_sha256"],
            "gate_config": vars(GateConfig()),
            "evaluated_as_of": candidate["trained_as_of"]}


def version_checks(out: Path, device: str, now: str) -> dict:
    manager = VersionManager(out, device)
    candidate = json.loads((out/"candidates"/"C0"/"manifest.json").read_text(encoding="utf-8"))
    checks = {"listing_before": manager.list_versions()}
    gate = controlled_gate(manager, candidate)
    checks["upgrade"] = manager.publish(candidate, gate, now, 0.90)
    checks["rollback"] = manager.activate("M0", now, "rollback")
    before = manager.current()
    try:
        manager.publish(candidate, gate, now, 0.90)  # gate bound to the pre-rollback activation
        raise AssertionError("stale gate was accepted")
    except ValueError as exc:
        checks["stale_gate_rejected"] = f"{type(exc).__name__}: {exc}"
    require(manager.current() == before, "pointer changed after stale gate rejection")
    checks["fresh_upgrade"] = manager.publish(candidate, controlled_gate(manager, candidate), now, 0.90)
    # The corruption check runs on a scratch copy so the kept artifact stays intact.
    scratch_root = Path(tempfile.mkdtemp())
    try:
        scratch = scratch_root/"scratch"
        (scratch/"versions").mkdir(parents=True)
        shutil.copytree(out/"versions"/"M0", scratch/"versions"/"M0")
        scratch_manager = VersionManager(scratch, device)
        scratch_manager.activate("M0", now, "bootstrap")
        (scratch/"versions"/"M0"/"M0_checkpoint.pt").write_bytes(b"broken")
        pointer = scratch_manager.current()
        try:
            scratch_manager.activate("M0", now, "rollback")
            raise AssertionError("corrupted checkpoint was accepted")
        except ValueError as exc:
            checks["corruption_rejected"] = f"{type(exc).__name__}: {exc}"
        require(scratch_manager.current() == pointer, "pointer changed after corruption rejection")
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)
    checks["listing_after"] = manager.list_versions()
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--version-checks", action="store_true",
                        help="exercise upgrade/rollback/stale-gate/corruption paths")
    args = parser.parse_args()
    import torch  # fail explicitly if PyTorch is absent
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {args.device} is unavailable")
    values, times = load_csv(Path(__file__).with_name("ETTh1.csv"))
    config = TrainConfig(backend="torch", model_name="LSTM", device=args.device,
                         epochs=1, patience=1, train_stride=24)
    summary = simulate(values[:3048], times[:3048], args.out, config,
                       train_end=1200, online_start=3000,
                       monitor_n=1, patience=2, max_episodes=2, exercise_at=2)
    require(not summary["candidate_errors"], f'Candidate failed: {summary["candidate_errors"]}')
    manager = VersionManager(args.out, args.device)
    restored = manager.activate("M0", summary["final_as_of"], "rollback")
    result = dict(torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
                  device=args.device, summary=summary, rollback=restored,
                  scope="execution smoke only; no claimed performance gain")
    if args.version_checks:
        result["version_checks"] = version_checks(Path(args.out), args.device, summary["final_as_of"])
    write_json(Path(args.out)/"torch_smoke.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
