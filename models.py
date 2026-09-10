import torch
from torch import nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    """Strict linear baseline: flattened history -> 24-step forecast."""
    def __init__(self, input_len, n_features, pred_len):
        super().__init__()
        self.fc = nn.Linear(input_len * n_features, pred_len)

    def forward(self, x):
        return self.fc(x.reshape(x.size(0), -1))

class MLPModel(nn.Module):
    def __init__(self, input_len, n_features, pred_len, hidden_size=256, num_layers=2, dropout=.1):
        super().__init__()
        layers=[nn.Flatten(), nn.Linear(input_len*n_features,hidden_size),nn.GELU()]
        for _ in range(max(0,num_layers-1)):
            layers += [nn.Dropout(dropout),nn.Linear(hidden_size,hidden_size),nn.GELU()]
        layers += [nn.Dropout(dropout),nn.Linear(hidden_size,pred_len)]
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class CNN1DModel(nn.Module):
    def __init__(self,input_len,n_features,pred_len,hidden_size=128,kernel_size=5,dropout=.1):
        super().__init__()
        pad=kernel_size//2
        self.conv=nn.Sequential(nn.Conv1d(n_features,hidden_size,kernel_size,padding=pad),nn.GELU(),
                                nn.BatchNorm1d(hidden_size),nn.Conv1d(hidden_size,hidden_size,kernel_size,padding=pad),
                                nn.GELU(),nn.AdaptiveAvgPool1d(1))
        self.head=nn.Sequential(nn.Flatten(),nn.Dropout(dropout),nn.Linear(hidden_size,pred_len))
    def forward(self,x): return self.head(self.conv(x.transpose(1,2)))

class LSTMModel(nn.Module):
    def __init__(self,input_len,n_features,pred_len,hidden_size=128,num_layers=2,dropout=.1):
        super().__init__()
        self.lstm=nn.LSTM(n_features,hidden_size,num_layers=num_layers,batch_first=True,
                          dropout=dropout if num_layers>1 else 0)
        self.head=nn.Linear(hidden_size,pred_len)
    def forward(self,x):
        y,_=self.lstm(x); return self.head(y[:,-1,:])

class Chomp1d(nn.Module):
    def __init__(self,chomp): super().__init__(); self.chomp=chomp
    def forward(self,x): return x[:,:,:-self.chomp] if self.chomp else x

class TemporalBlock(nn.Module):
    def __init__(self,cin,cout,kernel_size,dilation,dropout):
        super().__init__()
        pad=(kernel_size-1)*dilation
        self.net=nn.Sequential(
            nn.Conv1d(cin,cout,kernel_size,padding=pad,dilation=dilation),
            Chomp1d(pad),nn.GELU(),nn.Dropout(dropout),
            nn.Conv1d(cout,cout,kernel_size,padding=pad,dilation=dilation),
            Chomp1d(pad),nn.GELU(),nn.Dropout(dropout))
        self.down=nn.Conv1d(cin,cout,1) if cin!=cout else nn.Identity()
    def forward(self,x): return F.gelu(self.net(x)+self.down(x))

class TCNModel(nn.Module):
    def __init__(self,input_len,n_features,pred_len,channels=(64,64,128),kernel_size=5,dropout=.1):
        super().__init__()
        blocks=[]; cin=n_features
        for i,cout in enumerate(channels):
            blocks.append(TemporalBlock(cin,cout,kernel_size,2**i,dropout)); cin=cout
        self.tcn=nn.Sequential(*blocks)
        self.head=nn.Linear(cin,pred_len)
    def forward(self,x): return self.head(self.tcn(x.transpose(1,2))[:,:,-1])

MODEL_CLASSES={"Linear":LinearModel,"MLP":MLPModel,"CNN1D":CNN1DModel,"LSTM":LSTMModel,"TCN":TCNModel}
