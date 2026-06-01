from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

@dataclass(slots=True)
class OptimizerConfig:
    name: str = 'adam'
    lr: float = 0.001
    weight_decay: float = 0.0001

@dataclass(slots=True)
class ExperimentConfig:
    dataset: str
    manifest_path: str
    split_name: str = 'bearing_primary'
    window_length: int = 4096
    hop: int = 2048
    sample_rate: int = 42000
    crop_seconds: float = 8.0
    modalities: list[str] = field(default_factory=lambda: ['accel', 'audio'])
    feature_frontend: str = 'fft'
    encoder: str = 'none'
    classifier: str = 'xgboost'
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    seeds: list[int] = field(default_factory=lambda: [13])
    metrics: list[str] = field(default_factory=lambda: ['balanced_accuracy', 'auroc', 'auprc', 'f1', 'sensitivity', 'specificity'])
    batch_size: int = 64
    epochs: int = 15
    patience: int = 5
    hidden_dim: int = 64
    n_mels: int = 64
    n_fft: int = 512
    stft_hop: int = 128
    wavelet_scales: int = 64
    wavelet_width: int = 128
    wavelet_downsample: int = 4
    frontend_filters: int = 32
    vp_latent_dim: int = 32
    vp_frame_length: int = 128
    vp_frame_step: int = 128
    vp_encoder_decay: float = 0.9
    vp_spike_threshold: float = 0.02
    snn_steps: int = 32
    snn_hidden_dim: int = 128
    output_dir: str = 'outputs/experiments/default'
    use_primary_only: bool = True
    weighted_loss: bool = True
    save_predictions: bool = True

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

def _as_optimizer_config(value: dict[str, Any] | OptimizerConfig | None) -> OptimizerConfig:
    if isinstance(value, OptimizerConfig):
        return value
    if value is None:
        return OptimizerConfig()
    return OptimizerConfig(**value)

def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open('r', encoding='utf-8') as handle:
        raw = yaml.safe_load(handle)
    raw = dict(raw)
    raw['optimizer'] = _as_optimizer_config(raw.get('optimizer'))
    return ExperimentConfig(**raw)
