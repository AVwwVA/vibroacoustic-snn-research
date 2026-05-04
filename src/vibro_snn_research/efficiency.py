from __future__ import annotations
import pickle
from pathlib import Path
import tempfile
import time
from typing import Any
import numpy as np
import psutil
import torch
from torch import nn

def count_parameters(model: Any) -> int | None:
    if isinstance(model, nn.Module):
        return int(sum((param.numel() for param in model.parameters())))
    if hasattr(model, 'named_steps') and 'mlp' in model.named_steps:
        mlp = model.named_steps['mlp']
        return int(sum((weight.size for weight in mlp.coefs_)) + sum((bias.size for bias in mlp.intercepts_)))
    return None

def serialized_model_size_bytes(model: Any) -> int:
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        if isinstance(model, nn.Module):
            torch.save(model.state_dict(), tmp_path)
        else:
            with tmp_path.open('wb') as file_handle:
                pickle.dump(model, file_handle)
        return tmp_path.stat().st_size
    finally:
        tmp_path.unlink(missing_ok=True)

def _conv_macs(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> int:
    if isinstance(module, nn.Conv1d):
        kernel_ops = module.kernel_size[0] * module.in_channels // module.groups
    elif isinstance(module, nn.Conv2d):
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
    else:
        return 0
    return int(np.prod(output.shape) * kernel_ops)

def estimate_torch_macs(model: nn.Module, *sample_inputs: torch.Tensor) -> dict[str, float]:
    macs = 0
    nonzero = 0
    total = 0
    handles = []

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal macs, nonzero, total
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            macs += _conv_macs(module, inputs, output)
        elif isinstance(module, nn.Linear):
            macs += int(output.numel() * module.in_features)
        if isinstance(output, torch.Tensor) and output.is_floating_point():
            nonzero += int((output != 0).sum().item())
            total += int(output.numel())
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.ReLU, nn.AvgPool1d, nn.MaxPool1d, nn.MaxPool2d)):
            handles.append(module.register_forward_hook(hook))
    try:
        model.eval()
        with torch.no_grad():
            model(*sample_inputs)
    finally:
        for handle in handles:
            handle.remove()
    return {'macs': float(macs), 'flops': float(macs * 2), 'activation_density': float(nonzero / max(total, 1))}

def benchmark_torch_model(model: nn.Module, *sample_inputs: torch.Tensor, repeats: int=20) -> dict[str, float]:
    process = psutil.Process()
    model.eval()
    with torch.no_grad():
        model(*sample_inputs)
    start_mem = process.memory_info().rss
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            model(*sample_inputs)
    elapsed = time.perf_counter() - start
    end_mem = process.memory_info().rss
    stats = estimate_torch_macs(model, *sample_inputs)
    if hasattr(model, 'last_spike_stats'):
        stats.update(model.last_spike_stats)
    stats.update({'parameter_count': float(count_parameters(model) or 0), 'serialized_size_bytes': float(serialized_model_size_bytes(model)), 'latency_batch': float(elapsed / repeats), 'peak_ram_bytes': float(max(end_mem - start_mem, 0))})
    return stats

def benchmark_classical_model(model: Any, sample_features: np.ndarray, repeats: int=20) -> dict[str, float]:
    process = psutil.Process()
    start_mem = process.memory_info().rss
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict_proba(sample_features)
    elapsed = time.perf_counter() - start
    end_mem = process.memory_info().rss
    return {'parameter_count': float(count_parameters(model) or 0), 'serialized_size_bytes': float(serialized_model_size_bytes(model)), 'latency_batch': float(elapsed / repeats), 'peak_ram_bytes': float(max(end_mem - start_mem, 0)), 'macs': float('nan'), 'flops': float('nan'), 'activation_density': float('nan')}
