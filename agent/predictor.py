from pathlib import Path
import numpy as np, torch, yaml
from models import MODEL_CLASSES
from agent.risk_engine import summarize

class OilTemperatureAgent:
    def __init__(self,checkpoint,device="cuda:0"):
        ckpt=torch.load(checkpoint,map_location="cpu",weights_only=False)
        self.device=torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
        self.ckpt=ckpt; self.cfg=ckpt["config"]; self.model_name=ckpt["model_name"]
        mc=ckpt["model_config"]; c=self.cfg
        MODEL_KWARGS = {
            "Linear": set(),
            "MLP": {"hidden", "layers", "dropout"},
            "CNN1D": {"hidden", "kernel_size", "dropout"},
            "LSTM": {"hidden", "layers", "dropout"},
            "TCN": {"channels", "kernel_size", "dropout", "dilation_base"},
        }
        model_kwargs = {
            k: v for k, v in mc.items()
            if k in MODEL_KWARGS.get(self.model_name, set())
        }
        if self.model_name == "Linear":
            model_kwargs = {}
        elif self.model_name == "MLP":
            model_kwargs = {k: mc[k] for k in ("hidden", "layers", "dropout") if k in mc}
        elif self.model_name == "CNN1D":
            model_kwargs = {k: mc[k] for k in ("hidden", "kernel_size", "dropout") if k in mc}
        elif self.model_name == "LSTM":
            model_kwargs = {k: mc[k] for k in ("hidden", "layers", "dropout") if k in mc}
        elif self.model_name == "TCN":
            model_kwargs = {k: mc[k] for k in ("channels", "kernel_size", "dropout", "dilation_base") if k in mc}
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        self.model=MODEL_CLASSES[self.model_name](
            c["input_len"], len(c["features"]), c["pred_len"], **model_kwargs
        )
        self.model.load_state_dict(ckpt["model_state"]); self.model.to(self.device).eval()
        self.mean=np.array(ckpt["scaler_mean"]); self.scale=np.array(ckpt["scaler_scale"])
        self.target_idx=ckpt["target_idx"]
    def predict(self,history):
        arr=np.asarray(history,dtype=float)
        if arr.shape!=(self.cfg["input_len"],len(self.cfg["features"])):
            raise ValueError(f"history must have shape ({self.cfg['input_len']},{len(self.cfg['features'])})")
        z=(arr-self.mean)/self.scale
        with torch.no_grad():
            pred=self.model(torch.tensor(z,dtype=torch.float32).unsqueeze(0).to(self.device)).cpu().numpy()[0]
        forecast=pred*self.scale[self.target_idx]+self.mean[self.target_idx]
        risk_cfg=self.cfg["risk"]
        result={"model":self.model_name,"forecast_hours":len(forecast),
                "forecast":forecast.tolist()}
        result.update(summarize(forecast,risk_cfg["medium"],risk_cfg["high"]))
        return result
