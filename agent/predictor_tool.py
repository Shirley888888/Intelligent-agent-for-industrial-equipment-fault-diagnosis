from pathlib import Path
import pickle
import numpy as np
import torch
import yaml
from models import MODEL_CLASSES

class PredictorTool:
    """Validated 96h x 7-variable -> 24h OT forecasting tool."""

    def __init__(self, checkpoint="agent/models/best_model.pt",
                 scaler_path="agent/models/scaler.pkl",
                 config_path="agent/models/config.yaml", device="cuda:0"):
        self.checkpoint_path = Path(checkpoint)
        self.scaler_path = Path(scaler_path)
        self.config_path = Path(config_path)

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.ckpt = ckpt
        self.model_name = ckpt["model_name"]
        c = ckpt["config"]
        self.features = list(c["features"])
        self.target = c["target"]
        self.input_len = int(c["input_len"])
        self.pred_len = int(c["pred_len"])
        self.target_idx = int(ckpt["target_idx"])

        kwargs = self._model_kwargs(self.model_name, ckpt["model_config"])
        self.model = MODEL_CLASSES[self.model_name](
            self.input_len, len(self.features), self.pred_len, **kwargs
        )
        self.model.load_state_dict(ckpt["model_state"])

        use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
        self.device = torch.device(device if use_cuda else "cpu")
        self.model.to(self.device).eval()

        self.mean = np.asarray(self.scaler["mean"], dtype=float)
        self.scale = np.asarray(self.scaler["scale"], dtype=float)

        if self.mean.shape != (len(self.features),) or self.scale.shape != (len(self.features),):
            raise ValueError("Scaler dimensions do not match configured features.")

    @staticmethod
    def _model_kwargs(name, mc):
        if name == "Linear":
            return {}
        if name == "MLP":
            return {k: mc[k] for k in ("hidden_size","num_layers","dropout") if k in mc}
        if name == "CNN1D":
            return {k: mc[k] for k in ("hidden_size","kernel_size","dropout") if k in mc}
        if name == "LSTM":
            return {k: mc[k] for k in ("hidden_size","num_layers","dropout") if k in mc}
        if name == "TCN":
            return {k: mc[k] for k in ("channels","kernel_size","dropout","dilation_base") if k in mc}
        raise ValueError(f"Unsupported model: {name}")

    def validate(self, history):
        arr = np.asarray(history, dtype=float)
        expected = (self.input_len, len(self.features))
        if arr.shape != expected:
            raise ValueError(f"history must have shape {expected}, got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError("history contains NaN or infinite values")
        return arr

    def predict(self, history):
        arr = self.validate(history)
        z = (arr - self.mean) / self.scale
        x = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_z = self.model(x).detach().cpu().numpy()[0]
        forecast = pred_z * self.scale[self.target_idx] + self.mean[self.target_idx]
        return forecast.astype(float)

    def run(self, history):
        forecast = self.predict(history)
        return {
            "model": self.model_name,
            "input_hours": self.input_len,
            "forecast_hours": len(forecast),
            "target": self.target,
            "forecast": forecast.tolist()
        }

# Backward-compatible alias
OilTemperaturePredictorTool = PredictorTool
