"""5 个预测模型的 PyTorch 实现。

输入张量统一为 (batch, seq_len, n_features)，输出 (batch, pred_len)。
"""
import torch
import torch.nn as nn


class LinearModel(nn.Module):
    """线性回归：把整个输入窗口展平后直接线性映射到未来 24 小时。"""

    def __init__(self, input_len=96, n_features=7, pred_len=24):
        super().__init__()
        self.fc = nn.Linear(input_len * n_features, pred_len)

    def forward(self, x):
        return self.fc(x.reshape(x.shape[0], -1))


class MLP(nn.Module):
    """多层感知机：展平输入 -> 若干隐藏层(ReLU) -> 输出。"""

    def __init__(self, input_len=96, n_features=7, hidden=64, layers=2, pred_len=24):
        super().__init__()
        in_dim = input_len * n_features
        seq = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            seq += [nn.Linear(hidden, hidden), nn.ReLU()]
        seq.append(nn.Linear(hidden, pred_len))
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))


class Conv1DModel(nn.Module):
    """一维卷积：沿时间轴提取局部模式 -> 全局平均池化 -> 全连接输出。"""

    def __init__(self, in_channels=7, channels=32, kernel=3, pred_len=24):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, channels, kernel)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, pred_len)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class LSTMModel(nn.Module):
    """LSTM：沿时间步记忆时序依赖，取最后时间步的隐状态输出。"""

    def __init__(self, n_features=7, hidden=32, layers=1, pred_len=24):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden,
                            num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
