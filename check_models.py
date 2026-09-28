"""Constructor/forward smoke test for the four ETTh1 training candidates."""
import torch
import yaml
from pathlib import Path
from models import MODEL_CLASSES

ROOT = Path(__file__).resolve().parent
cfg = yaml.safe_load(open(ROOT / "configs/experiment.yaml", encoding="utf-8"))
x = torch.randn(2, cfg["input_len"], len(cfg["features"]))

for name, mc in cfg["models"].items():
    common = {"lr", "batch_size", "epochs", "patience"}
    kwargs = {k: v for k, v in mc.items() if k not in common}
    model = MODEL_CLASSES[name](cfg["input_len"], len(cfg["features"]), cfg["pred_len"], **kwargs)
    y = model(x)
    assert tuple(y.shape) == (2, cfg["pred_len"]), (name, y.shape)
    params = sum(p.numel() for p in model.parameters())
    print(f"{name:7s} OK params={params:,} output={tuple(y.shape)}")

print("FOUR-MODEL CONSTRUCTOR CHECK PASSED")
