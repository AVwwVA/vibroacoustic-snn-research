from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

class LearnableFilterBank1d(nn.Module):

    def __init__(self, n_filters: int=32, kernel_size: int=129, pool_stride: int=128) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, n_filters, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.pool = nn.AvgPool1d(kernel_size=pool_stride, stride=pool_stride)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.log1p(torch.abs(x))
        return self.pool(x)

class AdaptiveWaveletBank1d(nn.Module):

    def __init__(self, n_filters: int=32, kernel_size: int=129, pool_stride: int=128) -> None:
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=pool_stride, stride=pool_stride)
        self.log_scales = nn.Parameter(torch.linspace(math.log(0.015), math.log(0.15), n_filters))
        self.frequencies = nn.Parameter(torch.linspace(0.04, 0.35, n_filters))
        self.amplitudes = nn.Parameter(torch.ones(n_filters))
        self.register_buffer('time_axis', torch.linspace(-1.0, 1.0, kernel_size))

    def kernels(self) -> torch.Tensor:
        time_axis = self.time_axis[None, :]
        scales = torch.exp(self.log_scales)[:, None]
        frequencies = torch.clamp(self.frequencies[:, None], 0.01, 0.49)
        gauss = torch.exp(-time_axis ** 2 / (2.0 * scales ** 2))
        oscillation = torch.cos(2.0 * math.pi * frequencies * time_axis)
        kernels = self.amplitudes[:, None] * gauss * oscillation
        kernels = kernels - kernels.mean(dim=1, keepdim=True)
        norms = torch.linalg.norm(kernels, dim=1, keepdim=True).clamp_min(1e-06)
        return (kernels / norms).unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernels = self.kernels()
        x = F.conv1d(x, kernels, padding=self.kernel_size // 2)
        x = torch.log1p(torch.abs(x))
        return self.pool(x)

class AdaptiveFrontEndANNModel(nn.Module):

    def __init__(self, frontend: str='learnable_filterbank', n_filters: int=32, hidden_dim: int=64) -> None:
        super().__init__()
        frontend_cls = LearnableFilterBank1d if frontend == 'learnable_filterbank' else AdaptiveWaveletBank1d
        self.accel_frontend = frontend_cls(n_filters=n_filters)
        self.audio_frontend = frontend_cls(n_filters=n_filters)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n_filters * 2 * 32, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, 1))

    def sequence_features(self, accel: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        accel = self.accel_frontend(accel.unsqueeze(1))
        audio = self.audio_frontend(audio.unsqueeze(1))
        return torch.cat([accel, audio], dim=1)

    def forward(self, accel: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        features = self.sequence_features(accel, audio)
        if features.shape[-1] != 32:
            features = F.adaptive_avg_pool1d(features, 32)
        return self.head(features).squeeze(-1)

class SpikeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx: torch.Tensor, input_tensor: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.threshold = threshold
        return (input_tensor >= threshold).float()

    @staticmethod
    def backward(ctx: torch.Tensor, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        input_tensor, = ctx.saved_tensors
        mask = (input_tensor - ctx.threshold).abs() <= 0.5
        return (grad_output * mask.float(), None)

class LIFLayer(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, beta: float=0.9, threshold: float=1.0) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.beta = beta
        self.threshold = threshold

    def forward(self, currents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, steps = currents.shape
        membrane = torch.zeros(batch_size, self.linear.out_features, device=currents.device)
        spikes = []
        for step in range(steps):
            input_current = self.linear(currents[:, :, step])
            membrane = self.beta * membrane + input_current
            spike = SpikeFunction.apply(membrane, self.threshold)
            membrane = membrane * (1.0 - spike)
            spikes.append(spike)
        spike_tensor = torch.stack(spikes, dim=-1)
        return (spike_tensor, membrane)

class AdaptiveFrontEndSNNModel(nn.Module):

    def __init__(self, frontend: str='adaptive_wavelet', encoder: str='delta', n_filters: int=32, hidden_dim: int=128) -> None:
        super().__init__()
        frontend_cls = LearnableFilterBank1d if frontend == 'learnable_filterbank' else AdaptiveWaveletBank1d
        self.frontend_name = frontend
        self.encoder = encoder
        self.accel_frontend = frontend_cls(n_filters=n_filters)
        self.audio_frontend = frontend_cls(n_filters=n_filters)
        encoded_dim = n_filters * 4 if encoder == 'delta' else n_filters * 2
        self.lif_hidden = LIFLayer(encoded_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, 1)
        self.last_spike_stats: dict[str, float] = {}

    def _frontend(self, accel: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        accel = self.accel_frontend(accel.unsqueeze(1))
        audio = self.audio_frontend(audio.unsqueeze(1))
        features = torch.cat([accel, audio], dim=1)
        if features.shape[-1] != 32:
            features = F.adaptive_avg_pool1d(features, 32)
        return features

    @staticmethod
    def _normalize_features(features: torch.Tensor) -> torch.Tensor:
        mins = features.amin(dim=-1, keepdim=True)
        maxs = features.amax(dim=-1, keepdim=True)
        return (features - mins) / (maxs - mins + 1e-06)

    def _encode(self, features: torch.Tensor) -> torch.Tensor:
        if self.encoder == 'delta':
            deltas = features[:, :, 1:] - features[:, :, :-1]
            positive = (deltas > 0.02).float()
            negative = (deltas < -0.02).float()
            encoded = torch.cat([positive, negative], dim=1)
            return F.pad(encoded, (1, 0))
        normalized = self._normalize_features(features)
        return normalized

    def forward(self, accel: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        features = self._frontend(accel, audio)
        currents = self._encode(features)
        spikes, _ = self.lif_hidden(currents)
        spike_rate = spikes.mean(dim=-1)
        logits = self.readout(spike_rate).squeeze(-1)
        spike_count = float(spikes.sum().item())
        spike_density = float((spikes > 0).float().mean().item())
        synaptic_ops = float(spike_count * self.readout.in_features)
        self.last_spike_stats = {'spike_count': spike_count, 'spike_density': spike_density, 'estimated_synaptic_ops': synaptic_ops}
        return logits
