import yaml
from models import MODEL_CLASSES

with open("configs/experiment.yaml","r",encoding="utf-8") as f:
    cfg=yaml.safe_load(f)

MODEL_KWARGS = {
    "Linear": set(),
    "MLP": {"hidden","layers","dropout"},
    "CNN1D": {"hidden","kernel_size","dropout"},
    "LSTM": {"hidden","layers","dropout"},
    "TCN": {"channels","kernel_size","dropout","dilation_base"},
}

for name, mc in cfg["models"].items():
    kw={k:v for k,v in mc.items() if k in MODEL_KWARGS[name]}
    model=MODEL_CLASSES[name](cfg["input_len"],len(cfg["features"]),cfg["pred_len"],**kw)
    print(name, "OK", "params=", sum(p.numel() for p in model.parameters()), "kwargs=", kw)
print("ALL MODEL CONSTRUCTORS OK")
