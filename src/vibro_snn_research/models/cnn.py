from __future__ import annotations
import torch
from torch import nn

class _SpectrogramBranch(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 48, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.net(x), start_dim=1)

class TwoBranchSpectrogramCNN(nn.Module):

    def __init__(self, hidden_dim: int=64) -> None:
        super().__init__()
        self.accel_branch = _SpectrogramBranch()
        self.audio_branch = _SpectrogramBranch()
        self.head = nn.Sequential(nn.Linear(48 * 4 * 4 * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        accel = self.accel_branch(x[:, 0:1, :, :])
        audio = self.audio_branch(x[:, 1:2, :, :])
        fused = torch.cat([accel, audio], dim=1)
        return self.head(fused).squeeze(-1)

class RawSignalCNN(nn.Module):

    def __init__(self, hidden_dim: int=64) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv1d(2, 16, kernel_size=15, stride=2, padding=7), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(32, 48, kernel_size=5, stride=2, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(32))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(48 * 32, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, 1))

    def forward(self, accel: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        x = torch.stack([accel, audio], dim=1)
        return self.head(self.features(x)).squeeze(-1)
