from pathlib import Path
import pickle
import numpy as np
import torch
import yaml
from models import LSTMModel  # modify:model-unification - deployment has one model only.


class PredictorTool:
    """Validated 96h x 7-variable -> 24h OT forecasting tool using the configured LSTM only."""

    def __init__(self, checkpoint=None, scaler_path=None,
                 config_path="agent/models/config.yaml", device="cuda:0"):
        root = Path(__file__).resolve().parents[1]
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = root / self.config_path
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # modify:model-unification - model.type is the single source of truth.
        self.model_name = self.cfg["model"]["type"]
        if self.model_name != "LSTM":
            raise ValueError("This Agent deployment is unified to model.type=LSTM only.")
        self.checkpoint_path = Path(checkpoint or self.cfg["model_checkpoint"])
        self.scaler_path = Path(scaler_path or self.cfg["scaler"])
        if not self.checkpoint_path.is_absolute(): self.checkpoint_path = root / self.checkpoint_path
        if not self.scaler_path.is_absolute(): self.scaler_path = root / self.scaler_path
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if ckpt["model_name"] != self.model_name:
            raise ValueError(f"Checkpoint model={ckpt['model_name']} does not match config model.type={self.model_name}.")
        c = ckpt["config"]
        self.features, self.target = list(c["features"]), c["target"]
        self.input_len, self.pred_len = int(c["input_len"]), int(c["pred_len"])
        self.target_idx = int(ckpt["target_idx"])

        # modify:model-unification - no Linear/TCN/other initialization branches.
        mc = self.cfg["model"]
        self.model = LSTMModel(self.input_len, len(self.features), self.pred_len,
                               hidden_size=int(mc["hidden_size"]), num_layers=int(mc["num_layers"]),
                               dropout=float(mc["dropout"]))
        self.model.load_state_dict(ckpt["model_state"])
        use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
        self.device = torch.device(device if use_cuda else "cpu")
        self.model.to(self.device).eval()
        self.mean = np.asarray(self.scaler["mean"], dtype=float)
        self.scale = np.asarray(self.scaler["scale"], dtype=float)
        if self.mean.shape != (len(self.features),) or self.scale.shape != (len(self.features),):
            raise ValueError("Scaler dimensions do not match configured features.")

    def validate(self, history):
        arr = np.asarray(history, dtype=float)
        expected = (self.input_len, len(self.features))
        if arr.shape != expected: raise ValueError(f"history must have shape {expected}, got {arr.shape}")
        if not np.isfinite(arr).all(): raise ValueError("history contains NaN or infinite values")
        return arr

    def predict(self, history):
        arr = self.validate(history)
        z = (arr - self.mean) / self.scale
        x = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad(): pred_z = self.model(x).detach().cpu().numpy()[0]
        return (pred_z * self.scale[self.target_idx] + self.mean[self.target_idx]).astype(float)

    def run(self, history):
        forecast = self.predict(history)
        return {"model": self.model_name, "input_hours": self.input_len, "forecast_hours": len(forecast),
                "target": self.target, "forecast": forecast.tolist()}

OilTemperaturePredictorTool = PredictorTool
