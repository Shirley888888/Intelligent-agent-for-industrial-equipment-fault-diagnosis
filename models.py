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






class TrendResidualBlock(nn.Module):
    """Learnable trend/short-term residual recombination on the time axis."""
    def __init__(self, in_dim, hidden, kernel_size=5):
        super().__init__()
        self.project = nn.Linear(in_dim, hidden) if in_dim != hidden else nn.Identity()
        self.kernel_size = max(3, int(kernel_size) | 1)
        self.trend_gate = nn.Parameter(torch.tensor(0.5))
        self.residual_gate = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        x = self.project(x)
        k = min(self.kernel_size, x.shape[1] if x.shape[1] % 2 else max(1, x.shape[1]-1))
        if k < 1: return self.norm(x)
        trend = torch.nn.functional.avg_pool1d(
            x.transpose(1, 2), kernel_size=k, stride=1, padding=k//2, count_include_pad=False
        ).transpose(1, 2)
        trend = trend[:, :x.shape[1], :]
        residual = x - trend
        return self.norm(self.trend_gate * trend + self.residual_gate * residual + x)


class FrequencyMixerBlock(nn.Module):
    """Low/high-frequency temporal mixing using a real FFT decomposition."""
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.project = nn.Linear(in_dim, hidden) if in_dim != hidden else nn.Identity()
        self.low_gain = nn.Parameter(torch.ones(1, 1, hidden))
        self.high_gain = nn.Parameter(torch.ones(1, 1, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        x = self.project(x)
        spectrum = torch.fft.rfft(x.float(), dim=1, norm="ortho")
        split = max(1, spectrum.shape[1] // 4)
        low = torch.zeros_like(spectrum); low[:, :split, :] = spectrum[:, :split, :]
        high = spectrum - low
        mixed = torch.fft.irfft(low * self.low_gain + high * self.high_gain,
                                n=x.shape[1], dim=1, norm="ortho")
        return self.norm(x + mixed.to(dtype=x.dtype))


class AmplitudePhaseBlock(nn.Module):
    """Learnable spectral amplitude/phase operator for periodic and shifted patterns."""
    def __init__(self, in_dim, hidden, input_len=96):
        super().__init__()
        self.project = nn.Linear(in_dim, hidden) if in_dim != hidden else nn.Identity()
        self.log_amplitude = nn.Parameter(torch.zeros(1, int(input_len)//2 + 1, hidden))
        self.phase = nn.Parameter(torch.zeros(1, int(input_len)//2 + 1, hidden))
        self.residual_gate = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        x = self.project(x)
        spectrum = torch.fft.rfft(x.float(), dim=1, norm="ortho")
        amp = torch.nn.functional.interpolate(
            self.log_amplitude.transpose(1, 2), size=spectrum.shape[1],
            mode="linear", align_corners=True).transpose(1, 2).exp().clamp(0.1, 10.0)
        phase = torch.nn.functional.interpolate(
            self.phase.transpose(1, 2), size=spectrum.shape[1],
            mode="linear", align_corners=True).transpose(1, 2)
        rotation = torch.complex(torch.cos(phase), torch.sin(phase))
        transformed = torch.fft.irfft(spectrum * amp * rotation, n=x.shape[1], dim=1, norm="ortho")
        return self.norm(x + self.residual_gate * transformed.to(dtype=x.dtype))


class CrossVariableBlock(nn.Module):
    """Attention across observed variables at each time step, then pool variables."""
    def __init__(self, in_dim, hidden, heads=4, dropout=0.1):
        super().__init__()
        heads = next(h for h in (8, 4, 2, 1) if hidden % h == 0) if hidden % heads else heads
        self.value_proj = nn.Linear(1, hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.in_dim = int(in_dim)

    def forward(self, x):
        # Treat each input channel/latent channel as a token independently per timestamp.
        tokens = self.value_proj(x.unsqueeze(-1))  # [B,T,C,H]
        b, t, c, h = tokens.shape
        tokens = tokens.reshape(b*t, c, h)
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        pooled = self.norm((tokens + attended).mean(dim=1))
        return pooled.reshape(b, t, h)




# V6: safe meta-operator compiler. The agent searches declarative primitive
# graphs; it never writes/imports arbitrary Python code.
META_PRIMITIVES = ("layer_norm", "depthwise_conv", "pointwise_mlp",
                   "gated_mlp", "temporal_attention", "spectral_gain",
                   "trend_split", "channel_gate")

class _MetaPrimitive(nn.Module):
    def __init__(self, name, hidden, heads=4, dropout=0.1):
        super().__init__(); self.name=name; self.hidden=int(hidden)
        if name == "layer_norm": self.op=nn.LayerNorm(hidden)
        elif name == "depthwise_conv":
            self.op=nn.Conv1d(hidden, hidden, 5, padding=2, groups=hidden)
        elif name == "pointwise_mlp":
            self.op=nn.Sequential(nn.Linear(hidden, hidden*2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden*2, hidden))
        elif name == "gated_mlp":
            self.value=nn.Linear(hidden, hidden); self.gate=nn.Linear(hidden, hidden)
            self.out=nn.Linear(hidden, hidden)
        elif name == "temporal_attention":
            h=next(v for v in (8,4,2,1) if hidden % v == 0) if hidden % heads else heads
            self.op=nn.MultiheadAttention(hidden,h,dropout=dropout,batch_first=True)
        elif name == "spectral_gain":
            self.low=nn.Parameter(torch.ones(1,1,hidden)); self.high=nn.Parameter(torch.ones(1,1,hidden))
        elif name == "trend_split":
            self.trend_gate=nn.Parameter(torch.tensor(0.5)); self.residual_gate=nn.Parameter(torch.tensor(0.5))
            self.norm=nn.LayerNorm(hidden)
        elif name == "channel_gate":
            self.op=nn.Sequential(nn.Linear(hidden, max(4,hidden//4)), nn.GELU(),
                                  nn.Linear(max(4,hidden//4),hidden), nn.Sigmoid())
        else: raise ValueError(f"unregistered meta primitive: {name}")

    def forward(self,x):
        if self.name == "layer_norm": return self.op(x)
        if self.name == "depthwise_conv": return self.op(x.transpose(1,2)).transpose(1,2)
        if self.name == "pointwise_mlp": return self.op(x)
        if self.name == "gated_mlp": return self.out(self.value(x)*torch.sigmoid(self.gate(x)))
        if self.name == "temporal_attention":
            y,_=self.op(x,x,x,need_weights=False); return y
        if self.name == "spectral_gain":
            z=torch.fft.rfft(x.float(),dim=1,norm="ortho"); split=max(1,z.shape[1]//4)
            low=torch.zeros_like(z); low[:,:split]=z[:,:split]; high=z-low
            y=torch.fft.irfft(low*self.low+high*self.high,n=x.shape[1],dim=1,norm="ortho")
            return y.to(dtype=x.dtype)
        if self.name == "trend_split":
            k=min(9, x.shape[1] if x.shape[1]%2 else max(1,x.shape[1]-1))
            trend=torch.nn.functional.avg_pool1d(x.transpose(1,2),k,stride=1,padding=k//2,count_include_pad=False).transpose(1,2)
            trend=trend[:,:x.shape[1]]; return self.norm(self.trend_gate*trend+self.residual_gate*(x-trend))
        if self.name == "channel_gate": return x*self.op(x.mean(dim=1,keepdim=True))
        raise RuntimeError("unreachable meta primitive")

class MetaOperatorBlock(nn.Module):
    """Compiled bounded primitive graph; each discovered operation is residual."""
    def __init__(self,in_dim,hidden,ops,heads=4,dropout=0.1):
        super().__init__()
        if not isinstance(ops,list) or not 1 <= len(ops) <= 6 or any(o not in META_PRIMITIVES for o in ops):
            raise ValueError("invalid meta-operator graph")
        self.project=nn.Linear(in_dim,hidden) if in_dim!=hidden else nn.Identity()
        self.ops=nn.ModuleList([_MetaPrimitive(o,hidden,heads,dropout) for o in ops])
        self.norm=nn.LayerNorm(hidden)
    def forward(self,x):
        x=self.project(x)
        for op in self.ops:
            y=op(x)
            x=x+y if y.shape==x.shape else y
        return self.norm(x)


class DARTSSearchBlock(nn.Module):
    """Differentiable mixed operation: softmax architecture weights combine primitives."""
    def __init__(self, in_dim, hidden, ops, heads=4, dropout=0.1):
        super().__init__()
        if not isinstance(ops, list) or not 2 <= len(ops) <= 8 or len(set(ops)) != len(ops) or any(o not in META_PRIMITIVES for o in ops):
            raise ValueError("invalid DARTS primitive set")
        self.project = nn.Linear(in_dim, hidden) if in_dim != hidden else nn.Identity()
        self.ops = list(ops)
        self.candidates = nn.ModuleList([_MetaPrimitive(o, hidden, heads, dropout) for o in ops])
        # Architecture logits are optimized jointly with network weights.
        self.architecture_logits = nn.Parameter(torch.zeros(len(ops)))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        x = self.project(x)
        weights = torch.softmax(self.architecture_logits, dim=0)
        mixed = sum(weight * op(x) for weight, op in zip(weights, self.candidates))
        return self.norm(x + mixed)


class MathematicalOperatorBlock(nn.Module):
    """V7 compiler for a bounded mathematical equation and two primitive paths.

    Equations are explicit templates. No source code or arbitrary expression is
    evaluated; the Agent searches only the validated equation/primitive grammar.
    """
    EQUATIONS = {"residual_add", "gated_residual", "multiplicative_modulation",
                 "parallel_residual", "trend_spectral_mix"}

    def __init__(self, in_dim, hidden, equation, ops, gate_ops, heads=4, dropout=0.1):
        super().__init__()
        if equation not in self.EQUATIONS:
            raise ValueError(f"unregistered mathematical equation: {equation}")
        if not isinstance(ops, list) or not 1 <= len(ops) <= 6:
            raise ValueError("math operator main path must contain 1..6 primitives")
        if not isinstance(gate_ops, list) or len(gate_ops) > 4:
            raise ValueError("math operator gate path must contain 0..4 primitives")
        if equation in ("gated_residual", "multiplicative_modulation", "parallel_residual") and not gate_ops:
            raise ValueError("selected equation requires gate path")
        self.project = nn.Linear(in_dim, hidden) if in_dim != hidden else nn.Identity()
        self.main_path = nn.ModuleList([_MetaPrimitive(op, hidden, heads, dropout) for op in ops])
        self.gate_path = nn.ModuleList([_MetaPrimitive(op, hidden, heads, dropout) for op in gate_ops])
        self.equation = equation
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(hidden)

    @staticmethod
    def _run_path(x, modules):
        y = x
        for module in modules:
            update = module(y)
            y = y + update if update.shape == y.shape else update
        return y

    def forward(self, x):
        x = self.project(x)
        main = self._run_path(x, self.main_path)
        gate = self._run_path(x, self.gate_path) if len(self.gate_path) else None
        if self.equation == "residual_add":
            out = x + self.alpha * main
        elif self.equation == "gated_residual":
            out = x + torch.sigmoid(gate) * self.alpha * main
        elif self.equation == "multiplicative_modulation":
            out = x * (1.0 + torch.tanh(gate)) + self.alpha * main
        elif self.equation == "parallel_residual":
            out = x + self.alpha * main + self.beta * gate
        else:  # trend_spectral_mix: main path is spectral; gate path is trend/seasonal
            out = x + self.alpha * main + self.beta * gate
        return self.norm(out)


class NASComposedModel(nn.Module):
    """Declarative multi-branch NAS model; topology is supplied as a validated spec."""
    def __init__(self, input_len=96, n_features=7, pred_len=24, branches=None,
                 hidden=64, layers=2, attention="none", attention_heads=4,
                 fusion="concat", dropout=0.1, **kwargs):
        super().__init__()
        branches = branches or [["lstm", "lstm"]]
        self.input_proj = nn.Linear(n_features, hidden) if attention == "input" else nn.Identity()
        self.branches = nn.ModuleList()
        self.branch_dims = []
        for branch in branches:
            blocks = []
            in_dim = int(hidden if attention == "input" else n_features)
            for kind in branch:
                if isinstance(kind, dict) and kind.get("type") == "meta_operator":
                    blocks.append(MetaOperatorBlock(in_dim, hidden, kind["ops"], attention_heads, dropout))
                elif isinstance(kind, dict) and kind.get("type") == "darts_operator":
                    blocks.append(DARTSSearchBlock(in_dim, hidden, kind["ops"], attention_heads, dropout))
                elif isinstance(kind, dict) and kind.get("type") == "math_operator":
                    blocks.append(MathematicalOperatorBlock(in_dim, hidden, kind["equation"],
                        kind["ops"], kind["gate_ops"], attention_heads, dropout))
                elif kind == "cnn":
                    blocks.append(nn.Conv1d(in_dim, hidden, 3, padding=1))
                elif kind == "lstm":
                    blocks.append(nn.LSTM(in_dim, hidden, batch_first=True))
                elif kind == "transformer":
                    if in_dim != hidden:
                        blocks.append(nn.Linear(in_dim, hidden))
                    if hidden % attention_heads:
                        raise ValueError("hidden must be divisible by attention_heads")
                    blocks.append(nn.TransformerEncoderLayer(
                        d_model=hidden, nhead=attention_heads, batch_first=True,
                        dropout=dropout, norm_first=True, activation="gelu"))
                elif kind == "linear":
                    blocks.append(nn.Linear(in_dim, hidden))
                elif kind == "trend_residual":
                    blocks.append(TrendResidualBlock(in_dim, hidden))
                elif kind == "frequency_mixer":
                    blocks.append(FrequencyMixerBlock(in_dim, hidden))
                elif kind == "cross_var":
                    blocks.append(CrossVariableBlock(in_dim, hidden, attention_heads, dropout))
                elif kind == "amplitude_phase":
                    blocks.append(AmplitudePhaseBlock(in_dim, hidden, input_len=input_len))
                else:
                    raise ValueError(f"unsupported NAS block: {kind}")
                in_dim = hidden
            self.branches.append(nn.ModuleList(blocks))
            self.branch_dims.append(hidden)
        self.attention_position = attention
        self.attentions = nn.ModuleList()
        if attention != "none":
            if hidden % attention_heads:
                raise ValueError("hidden must be divisible by attention_heads")
            count = (sum(len(b) + (1 if b and b[0] == "transformer" and attention != "input" else 0)
                         for b in branches) if attention == "after_each" else 1)
            self.attentions = nn.ModuleList([
                nn.MultiheadAttention(hidden, attention_heads, dropout=dropout, batch_first=True)
                for _ in range(count)])
        self.fusion = fusion
        n_branches = len(branches)
        if fusion == "concat":
            fusion_dim = hidden * n_branches
            self.fusion_gate = None
        elif fusion == "mean":
            fusion_dim = hidden
            self.fusion_gate = None
        elif fusion == "gated":
            fusion_dim = hidden
            self.fusion_gate = nn.Sequential(nn.Linear(hidden*n_branches, n_branches), nn.Softmax(dim=-1))
        else:
            raise ValueError(f"unsupported fusion: {fusion}")
        self.norm = nn.LayerNorm(fusion_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(fusion_dim, pred_len))

    def _encode_branch(self, x, blocks, attn_iter):
        h = x
        for block in blocks:
            if isinstance(block, nn.Conv1d):
                h = torch.relu(block(h.transpose(1, 2)).transpose(1, 2))
            elif isinstance(block, nn.LSTM):
                h, _ = block(h)
            else:
                h = block(h)
            if self.attention_position == "after_each":
                a, _ = self.attentions[next(attn_iter)](h, h, h, need_weights=False)
                h = h + a
        if self.attention_position == "branch_end":
            a, _ = self.attentions[0](h, h, h, need_weights=False)
            h = h + a
        return h[:, -1, :]

    def forward(self, x):
        import itertools
        attn_iter = iter(range(len(self.attentions)))
        branch_input = self.input_proj(x)
        if self.attention_position == "input":
            a, _ = self.attentions[0](branch_input, branch_input, branch_input, need_weights=False)
            branch_input = branch_input + a
        vectors = [self._encode_branch(branch_input, b, attn_iter) for b in self.branches]
        if self.attention_position == "fusion":
            stack = torch.stack(vectors, dim=1)
            a, _ = self.attentions[0](stack, stack, stack, need_weights=False)
            vectors = list((stack + a).unbind(dim=1))
        if self.fusion == "concat":
            fused = torch.cat(vectors, dim=-1)
        elif self.fusion == "mean":
            fused = torch.stack(vectors, dim=0).mean(dim=0)
        else:
            joined = torch.cat(vectors, dim=-1)
            weights = self.fusion_gate(joined)
            fused = sum(v * weights[:, i:i+1] for i, v in enumerate(vectors))
        return self.head(self.norm(fused))


# modify:four-model-training - four complementary ETTh1 candidate models.
MODEL_CLASSES = {
    "Linear": LinearModel,
    "CNN1D": CNN1DModel,
    "LSTM": LSTMModel,
    "TCN": TCNModel,
}


class LSTMAttentionModel(nn.Module):
    """LSTM encoder with self-attention refinement over the history window."""
    def __init__(self, input_len=96, n_features=7, pred_len=24, hidden=96,
                 layers=2, heads=4, dropout=0.1, **kwargs):
        super().__init__()
        hidden = int(hidden)
        heads = int(heads)
        if hidden % heads:
            raise ValueError("hidden must be divisible by attention heads")
        self.encoder = nn.LSTM(n_features, hidden, num_layers=max(1, int(layers)),
                               batch_first=True, dropout=dropout if int(layers) > 1 else 0.0)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, pred_len))

    def forward(self, x):
        h, _ = self.encoder(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        return self.head(self.norm(h + a)[:, -1])


class RevINLSTMModel(nn.Module):
    """Per-instance reversible normalization around an LSTM forecaster."""
    def __init__(self, input_len=96, n_features=7, pred_len=24, hidden=96,
                 layers=2, dropout=0.1, target_index=6, eps=1e-5, **kwargs):
        super().__init__()
        if not 0 <= int(target_index) < int(n_features):
            raise ValueError("target_index must identify an input feature")
        self.target_index = int(target_index)
        self.eps = float(eps)
        self.encoder = nn.LSTM(n_features, int(hidden), num_layers=max(1, int(layers)),
                               batch_first=True, dropout=dropout if int(layers) > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(int(hidden), pred_len))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True).detach()
        std = x.var(dim=1, keepdim=True, unbiased=False).add(self.eps).sqrt().detach()
        z = (x - mean) / std
        h, _ = self.encoder(z)
        y = self.head(h[:, -1])
        return y * std[:, 0, self.target_index:self.target_index + 1] + mean[:, 0, self.target_index:self.target_index + 1]


class PatchTransformerLite(nn.Module):
    """Patch-based transformer encoder for multivariate time-series windows."""
    def __init__(self, input_len=96, n_features=7, pred_len=24, d_model=96,
                 patch_len=16, stride=8, heads=4, layers=2, dropout=0.1, **kwargs):
        super().__init__()
        patch_len, stride, d_model, heads = int(patch_len), int(stride), int(d_model), int(heads)
        if patch_len < 2 or stride < 1 or d_model % heads:
            raise ValueError("invalid patch_len/stride or d_model must be divisible by heads")
        self.patch = nn.Conv1d(n_features, d_model, kernel_size=patch_len, stride=stride)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True,
            norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(layers)))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x):
        z = self.patch(x.transpose(1, 2)).transpose(1, 2)
        z = self.encoder(z)
        return self.head(self.norm(z).mean(dim=1))


class DecompLinearModel(nn.Module):
    """Moving-average trend/seasonal decomposition with separate linear heads."""
    def __init__(self, input_len=96, n_features=7, pred_len=24, moving_avg=25, **kwargs):
        super().__init__()
        k = max(3, int(moving_avg) | 1)
        self.input_len = int(input_len)
        self.target_index = int(kwargs.get("target_index", 6))
        self.avg = nn.AvgPool1d(kernel_size=k, stride=1, padding=k // 2, count_include_pad=False)
        self.trend = nn.Linear(self.input_len, int(pred_len))
        self.seasonal = nn.Linear(self.input_len, int(pred_len))

    def forward(self, x):
        target = x[:, :, self.target_index:self.target_index + 1].transpose(1, 2)
        trend = self.avg(target).squeeze(1)
        seasonal = target.squeeze(1) - trend
        return self.trend(trend) + self.seasonal(seasonal)


# Architecture candidates are additive: the original four model names and
# checkpoint formats remain unchanged.
MODEL_CLASSES.update({
    "LSTM_Attention": LSTMAttentionModel,
    "RevIN_LSTM": RevINLSTMModel,
    "PatchTransformerLite": PatchTransformerLite,
    "DecompLinear": DecompLinearModel,
})

MODEL_CLASSES["NAS_Composed"] = NASComposedModel
