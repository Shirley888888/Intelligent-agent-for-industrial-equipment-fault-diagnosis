import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, input_len=96, n_features=7, pred_len=24):
        super().__init__()
        self.fc = nn.Linear(input_len * n_features, pred_len)
    def forward(self, x):
        # x: batch, T, F
        b = x.shape[0]
        x = x.reshape(b, -1)
        return self.fc(x)

class MLP(nn.Module):
    def __init__(self, input_len=96, n_features=7, hidden=64, n_hidden_layers=2, pred_len=24):
        super().__init__()
        in_dim = input_len * n_features
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(n_hidden_layers-1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, pred_len))
        self.net = nn.Sequential(*layers)
    def forward(self,x):
        b = x.shape[0]
        x = x.reshape(b, -1)
        return self.net(x)

class Conv1DModel(nn.Module):
    def __init__(self, in_channels=7, out_channels=32, kernel_size=5, pred_len=24):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        # simple global pooling + fc
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_channels, pred_len)
    def forward(self,x):
        # x: batch, T, F  --> conv1d expects (batch, C, L)
        x = x.permute(0,2,1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class LSTMModel(nn.Module):
    def __init__(self, n_features=7, hidden_size=32, num_layers=1, pred_len=24):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self,x):
        # x: batch, T, F
        out, (h,c) = self.lstm(x)
        # use last time step
        last = out[:, -1, :]
        return self.fc(last)
