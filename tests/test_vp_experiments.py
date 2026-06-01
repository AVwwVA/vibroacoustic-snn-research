from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import savemat
import yaml
from vibro_snn_research.experiments import run_experiment

def _make_signal(amplitude: float, frequency: float, length: int=4096) -> np.ndarray:
    time = np.linspace(0.0, 1.0, length, endpoint=False, dtype=np.float32)
    accel = amplitude * np.sin(2.0 * np.pi * frequency * time)
    audio = 0.8 * amplitude * np.sin(2.0 * np.pi * (frequency + 3.0) * time)
    speed = np.full(length, 1.0, dtype=np.float32)
    load = np.full(length, 0.5, dtype=np.float32)
    return np.stack([accel, audio, speed, load], axis=1).astype(np.float32)

def _write_record(path: Path, record_id: str, amplitude: float, frequency: float) -> None:
    savemat(path, {record_id: _make_signal(amplitude=amplitude, frequency=frequency)})

def _build_manifest(tmp_path: Path) -> Path:
    records = [
        ('H_1_0', 1, 'healthy', 0, 'train', 'primary', 1, 0.8, 5.0),
        ('I_2_1', 2, 'inner_race', 1, 'train', 'primary', 1, 1.6, 19.0),
        ('H_4_0', 4, 'healthy', 0, 'val', 'primary', 1, 0.9, 6.0),
        ('I_9_1', 9, 'inner_race', 1, 'val', 'primary', 1, 1.7, 20.0),
        ('H_5_0', 5, 'healthy', 0, 'test', 'primary', 1, 1.0, 7.0),
        ('I_10_1', 10, 'inner_race', 1, 'test', 'primary', 1, 1.8, 21.0),
        ('H_11_0', 11, 'healthy', 0, 'secondary', 'secondary', 0, 1.1, 8.0),
        ('B_11_1', 11, 'ball', 1, 'secondary', 'secondary', 0, 1.9, 22.0),
    ]
    rows = []
    for record_id, bearing_id, fault_family, health_state, split, subset, primary_included, amplitude, frequency in records:
        path = tmp_path / f'{record_id}.mat'
        _write_record(path, record_id, amplitude=amplitude, frequency=frequency)
        rows.append({'record_id': record_id, 'bearing_id': bearing_id, 'fault_family': fault_family, 'health_state': health_state, 'health_stage': 'healthy' if health_state == 0 else 'developing', 'binary_label': int(health_state > 0), 'split': split, 'subset': subset, 'primary_included': primary_included, 'path': str(path)})
    manifest_path = tmp_path / 'manifest.csv'
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path

def _write_config(tmp_path: Path, manifest_path: Path, name: str, classifier: str, encoder: str) -> Path:
    config = {
        'dataset': 'synthetic',
        'manifest_path': str(manifest_path),
        'split_name': 'bearing_primary',
        'window_length': 4096,
        'hop': 2048,
        'sample_rate': 4096,
        'crop_seconds': 1.0,
        'modalities': ['accel', 'audio'],
        'feature_frontend': 'vp_projection',
        'encoder': encoder,
        'classifier': classifier,
        'optimizer': {'name': 'adam', 'lr': 0.001, 'weight_decay': 0.0001},
        'seeds': [7],
        'metrics': ['balanced_accuracy', 'auroc', 'auprc', 'f1', 'sensitivity', 'specificity'],
        'batch_size': 2,
        'epochs': 1,
        'patience': 1,
        'hidden_dim': 16,
        'frontend_filters': 32,
        'vp_latent_dim': 32,
        'vp_frame_length': 128,
        'vp_frame_step': 128,
        'vp_encoder_decay': 0.9,
        'vp_spike_threshold': 0.02,
        'snn_hidden_dim': 16,
        'output_dir': str(tmp_path / name),
        'use_primary_only': True,
    }
    config_path = tmp_path / f'{name}.yaml'
    with config_path.open('w', encoding='utf-8') as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config_path

def test_vp_experiment_smoke(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    for name, classifier, encoder in [('vp_ann_smoke', 'ann', 'none'), ('vp_snn_smoke', 'snn', 'delta')]:
        config_path = _write_config(tmp_path, manifest_path, name, classifier, encoder)
        run_experiment(config_path)
        output_dir = tmp_path / name
        assert (output_dir / 'summary.json').exists()
        seed_dir = output_dir / 'seed_7'
        assert (seed_dir / 'metrics.json').exists()
        assert (seed_dir / 'predictions.csv').exists()
        assert (seed_dir / 'efficiency.json').exists()
