"""Backward-compatible predictor entry point for the unified LSTM Agent deployment."""
from agent.predictor_tool import PredictorTool  # modify:model-unification


class OilTemperatureAgent(PredictorTool):
    # modify:model-unification - legacy class now delegates to the single LSTM PredictorTool.
    def __init__(self, checkpoint=None, device="cuda:0", config_path="agent/models/config.yaml", scaler_path=None):
        super().__init__(checkpoint=checkpoint, scaler_path=scaler_path, config_path=config_path, device=device)
