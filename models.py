"""Forecasting models for ETTh1 candidate-model training.

All models use input shape (batch, input_len, n_features) and return
(batch, pred_len).  The Agent deployment remains single-model; this module
contains the four candidate architectures used during comparative training.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearModel(nn.Module):
    """Strong linear baseline over the complete 96x7 history window."""

    def __init__(self, input_len=96, n_features=7, pred_len=24, **kwargs):
        super().__init__()
        self.fc = nn.Linear(input_len * n_features, pred_len)

    def forward(self, x):
        return self.fc(x.reshape(x.shape[0], -1))


class CNN1DModel(nn.Module):
    """1-D CNN for local temporal patterns followed by adaptive pooling."""

    def __init__(self, input_len=96, n_features=7, pred_len=24,
                 channels=128, kernel_size=5, dropout=0.1, **kwargs):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(channels, pred_len)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.net(x).squeeze(-1)
        return self.fc(x)


class LSTMModel(nn.Module):
    """LSTM for recurrent long-range temporal dependency modelling."""

    # Keep compatibility with the project's earlier LSTM checkpoints.
    def load_state_dict(self, state_dict, strict=True, **kwargs):
        state_dict = state_dict.copy()
        for suffix in ("weight", "bias"):
            old, new = "head." + suffix, "fc." + suffix
            if old in state_dict and new not in state_dict:
                state_dict[new] = state_dict.pop(old)
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def __init__(self, input_len=96, n_features=7, pred_len=24,
                 hidden=128, layers=2, dropout=0.1,
                 hidden_size=None, num_layers=None, **kwargs):
        super().__init__()
        if hidden_size is not None:
            hidden = hidden_size
        if num_layers is not None:
            layers = num_layers
        layers = max(1, int(layers))
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TemporalBlock(nn.Module):
    """Residual dilated causal-style temporal block used by TCNModel."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.padding = padding
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def _trim(self, x):
        return x[:, :, :-self.padding] if self.padding > 0 else x

    def forward(self, x):
        y = self.drop(self.act(self._trim(self.conv1(x))))
        y = self.drop(self.act(self._trim(self.conv2(y))))
        return self.act(y + self.residual(x))


class TCNModel(nn.Module):
    """Dilated residual TCN for long receptive fields and parallel training."""

    def __init__(self, input_len=96, n_features=7, pred_len=24,
                 channels=(64, 64, 128), kernel_size=5, dropout=0.1, **kwargs):
        super().__init__()
        if isinstance(channels, int):
            channels = (channels,)
        blocks = []
        in_ch = n_features
        for i, out_ch in enumerate(channels):
            blocks.append(TemporalBlock(in_ch, int(out_ch), int(kernel_size), 2 ** i, float(dropout)))
            in_ch = int(out_ch)
        self.tcn = nn.Sequential(*blocks)
        self.fc = nn.Linear(in_ch, pred_len)

    def forward(self, x):
        y = self.tcn(x.transpose(1, 2))
        return self.fc(y[:, :, -1])


# modify:four-model-training - four complementary ETTh1 candidate models.
MODEL_CLASSES = {
    "Linear": LinearModel,
    "CNN1D": CNN1DModel,
    "LSTM": LSTMModel,
    "TCN": TCNModel,
}
