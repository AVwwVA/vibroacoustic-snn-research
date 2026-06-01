from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
from .plotting import configure_headless_matplotlib
configure_headless_matplotlib()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch
from .config import ExperimentConfig, load_experiment_config
from .dataset import WindowDataset, build_window_index, dataset_to_numpy, load_manifest
from .efficiency import benchmark_classical_model, benchmark_torch_model
from .features import FFTFeatureExtractor, MelSpectrogramExtractor, WaveletScalogramExtractor
from .manifest import build_manifest
from .metrics import compute_binary_metrics, summarize_metric_runs
from .models import AdaptiveFrontEndANNModel, AdaptiveFrontEndSNNModel, RawSignalCNN, TwoBranchSpectrogramCNN, VPANNModel, VPSNNModel
from .training import make_raw_loader, make_tensor_loader, predict_classical_model, predict_torch_model, set_seed, train_classical_model, train_torch_model

def _ensure_manifest(config: ExperimentConfig) -> Path:
    manifest_path = Path(config.manifest_path)
    if manifest_path.exists():
        return manifest_path
    zip_path = Path(config.dataset)
    if not zip_path.exists():
        raise FileNotFoundError(f'Could not find dataset or manifest at {config.dataset}')
    build_manifest(zip_path=zip_path, output_dir='data/uods_vafdc/raw', manifest_path=manifest_path)
    return manifest_path

def _apply_modalities(arrays: dict[str, Any], modalities: list[str]) -> dict[str, Any]:
    arrays = dict(arrays)
    if 'accel' not in modalities:
        arrays['accel'] = np.zeros_like(arrays['accel'])
    if 'audio' not in modalities:
        arrays['audio'] = np.zeros_like(arrays['audio'])
    return arrays

def _prediction_frame(metadata: list[dict[str, Any]], labels: np.ndarray, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    rows = []
    preds = (scores >= threshold).astype(int)
    for meta, label, score, pred in zip(metadata, labels, scores, preds):
        rows.append({**meta, 'label': int(label), 'score': float(score), 'prediction': int(pred)})
    return pd.DataFrame(rows)

def _save_confusion_matrix(labels: np.ndarray, scores: np.ndarray, threshold: float, output_path: Path) -> None:
    preds = (scores >= threshold).astype(int)
    matrix = confusion_matrix(labels, preds, labels=[0, 1])
    plt.figure(figsize=(4, 4))
    plt.imshow(matrix, cmap='Blues')
    plt.xticks([0, 1], ['normal', 'abnormal'])
    plt.yticks([0, 1], ['normal', 'abnormal'])
    for row_idx in range(2):
        for col_idx in range(2):
            plt.text(col_idx, row_idx, str(matrix[row_idx, col_idx]), ha='center', va='center')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

def _severity_analysis(predictions: pd.DataFrame) -> dict[str, float]:
    abnormal = predictions.loc[predictions['label'] == 1]
    if abnormal.empty:
        return {}
    developing = abnormal.loc[abnormal['health_state'] == 1, 'score'].to_numpy()
    faulty = abnormal.loc[abnormal['health_state'] == 2, 'score'].to_numpy()
    result = {'developing_mean_score': float(developing.mean()) if len(developing) else float('nan'), 'faulty_mean_score': float(faulty.mean()) if len(faulty) else float('nan')}
    if len(developing) and len(faulty):
        severity_labels = np.concatenate([np.zeros(len(developing)), np.ones(len(faulty))])
        severity_scores = np.concatenate([developing, faulty])
        result['severity_auroc'] = float(roc_auc_score(severity_labels, severity_scores))
    return result

def _secondary_dataset(config: ExperimentConfig) -> WindowDataset:
    manifest = load_manifest(config.manifest_path)
    secondary = manifest.loc[manifest['subset'] == 'secondary'].copy()
    return WindowDataset(secondary, split='secondary', window_length=config.window_length, hop=config.hop, sample_rate=config.sample_rate, crop_seconds=config.crop_seconds, primary_only=False)

def _run_classical_fft(config: ExperimentConfig, output_dir: Path) -> dict[str, Any]:
    train_ds = WindowDataset(config.manifest_path, 'train', config.window_length, config.hop, config.sample_rate, config.crop_seconds, config.use_primary_only)
    val_ds = WindowDataset(config.manifest_path, 'val', config.window_length, config.hop, config.sample_rate, config.crop_seconds, config.use_primary_only)
    test_ds = WindowDataset(config.manifest_path, 'test', config.window_length, config.hop, config.sample_rate, config.crop_seconds, config.use_primary_only)
    train_np = dataset_to_numpy(train_ds)
    val_np = dataset_to_numpy(val_ds)
    test_np = dataset_to_numpy(test_ds)
    train_np = _apply_modalities(train_np, config.modalities)
    val_np = _apply_modalities(val_np, config.modalities)
    test_np = _apply_modalities(test_np, config.modalities)
    extractor = FFTFeatureExtractor(sample_rate=config.sample_rate)
    extractor.fit(None)
    train_x = extractor.transform(train_np)
    val_x = extractor.transform(val_np)
    test_x = extractor.transform(test_np)
    model, threshold = train_classical_model(config.classifier, train_x, train_np['labels'], val_x, val_np['labels'], config.seeds[0])
    val_scores = predict_classical_model(model, val_x)
    test_scores = predict_classical_model(model, test_x)
    val_metrics = compute_binary_metrics(val_np['labels'], val_scores, threshold)
    test_metrics = compute_binary_metrics(test_np['labels'], test_scores, threshold)
    predictions = _prediction_frame(test_np['metadata'], test_np['labels'], test_scores, threshold)
    efficiency = benchmark_classical_model(model, test_x[:64])
    _save_confusion_matrix(test_np['labels'], test_scores, threshold, output_dir / 'confusion_matrix.png')
    severity = _severity_analysis(predictions)
    secondary_ds = _secondary_dataset(config)
    secondary_np = dataset_to_numpy(secondary_ds)
    secondary_np = _apply_modalities(secondary_np, config.modalities)
    secondary_x = extractor.transform(secondary_np)
    secondary_scores = predict_classical_model(model, secondary_x)
    secondary_metrics = compute_binary_metrics(secondary_np['labels'], secondary_scores, threshold)
    secondary_predictions = _prediction_frame(secondary_np['metadata'], secondary_np['labels'], secondary_scores, threshold)
    secondary_predictions.to_csv(output_dir / 'secondary_predictions.csv', index=False)
    predictions.to_csv(output_dir / 'predictions.csv', index=False)
    (output_dir / 'metrics.json').write_text(json.dumps({'val': val_metrics, 'test': test_metrics, 'secondary': secondary_metrics, 'severity': severity}, indent=2), encoding='utf-8')
    (output_dir / 'efficiency.json').write_text(json.dumps(efficiency, indent=2), encoding='utf-8')
    return {'val': val_metrics, 'test': test_metrics, 'secondary': secondary_metrics, 'severity': severity, 'efficiency': efficiency}

def _materialize_tf_dataset(dataset: WindowDataset, extractor: Any, modalities: list[str]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    arrays = _apply_modalities(dataset_to_numpy(dataset), modalities)
    features = extractor.transform(arrays).astype(np.float32)
    return (features, arrays['labels'], arrays['metadata'])

def _feature_cache_path(config: ExperimentConfig, split_name: str) -> Path:
    signature = {'manifest_path': config.manifest_path, 'split_name': config.split_name, 'window_length': config.window_length, 'hop': config.hop, 'sample_rate': config.sample_rate, 'crop_seconds': config.crop_seconds, 'modalities': config.modalities, 'feature_frontend': config.feature_frontend, 'n_fft': config.n_fft, 'stft_hop': config.stft_hop, 'n_mels': config.n_mels, 'wavelet_scales': config.wavelet_scales, 'wavelet_width': config.wavelet_width, 'wavelet_downsample': config.wavelet_downsample, 'use_primary_only': config.use_primary_only, 'split': split_name}
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode('utf-8')).hexdigest()[:16]
    cache_root = Path('outputs/feature_cache') / config.feature_frontend
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / f'{split_name}_{digest}.npz'

def _load_or_materialize_tf_dataset(dataset: WindowDataset, extractor: Any, modalities: list[str], config: ExperimentConfig, split_name: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    cache_path = _feature_cache_path(config, split_name)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        metadata = [dict(item) for item in cached['metadata'].tolist()]
        return (cached['features'].astype(np.float32), cached['labels'].astype(np.int64), metadata)
    features, labels, metadata = _materialize_tf_dataset(dataset, extractor, modalities)
    np.savez_compressed(cache_path, features=features.astype(np.float32), labels=labels.astype(np.int64), metadata=np.asarray(metadata, dtype=object))
    return (features, labels, metadata)

def _train_torch_feature_model(model: torch.nn.Module, extractor: Any, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, metadata: list[dict[str, Any]], config: ExperimentConfig, output_dir: Path) -> dict[str, Any]:
    train_loader = make_tensor_loader(train_x, train_y, config.batch_size, shuffle=True)
    val_loader = make_tensor_loader(val_x, val_y, config.batch_size, shuffle=False)
    test_loader = make_tensor_loader(test_x, test_y, config.batch_size, shuffle=False)
    model, threshold, history = train_torch_model(model=model, train_loader=train_loader, val_loader=val_loader, train_labels=train_y, optimizer_config=config.optimizer, epochs=config.epochs, patience=config.patience)
    val_labels, val_scores = predict_torch_model(model, val_loader)
    test_labels, test_scores = predict_torch_model(model, test_loader)
    val_metrics = compute_binary_metrics(val_labels, val_scores, threshold)
    test_metrics = compute_binary_metrics(test_labels, test_scores, threshold)
    predictions = _prediction_frame(metadata, test_labels, test_scores, threshold)
    efficiency = benchmark_torch_model(model, torch.from_numpy(test_x[:64]).float())
    _save_confusion_matrix(test_labels, test_scores, threshold, output_dir / 'confusion_matrix.png')
    severity = _severity_analysis(predictions)
    predictions.to_csv(output_dir / 'predictions.csv', index=False)
    secondary_ds = _secondary_dataset(config)
    secondary_x, secondary_y, secondary_metadata = _load_or_materialize_tf_dataset(secondary_ds, extractor, config.modalities, config, 'secondary')
    secondary_loader = make_tensor_loader(secondary_x, secondary_y, config.batch_size, shuffle=False)
    sec_labels, sec_scores = predict_torch_model(model, secondary_loader)
    secondary_metrics = compute_binary_metrics(sec_labels, sec_scores, threshold)
    secondary_predictions = _prediction_frame(secondary_metadata, sec_labels, sec_scores, threshold)
    secondary_predictions.to_csv(output_dir / 'secondary_predictions.csv', index=False)
    (output_dir / 'metrics.json').write_text(json.dumps({'history': history, 'val': val_metrics, 'test': test_metrics, 'secondary': secondary_metrics, 'severity': severity}, indent=2), encoding='utf-8')
    (output_dir / 'efficiency.json').write_text(json.dumps(efficiency, indent=2), encoding='utf-8')
    return {'val': val_metrics, 'test': test_metrics, 'secondary': secondary_metrics, 'severity': severity, 'efficiency': efficiency}

def _train_torch_raw_model(model: torch.nn.Module, train_np: dict[str, Any], val_np: dict[str, Any], test_np: dict[str, Any], config: ExperimentConfig, output_dir: Path) -> dict[str, Any]:
    train_loader = make_raw_loader(train_np, config.batch_size, shuffle=True)
    val_loader = make_raw_loader(val_np, config.batch_size, shuffle=False)
    test_loader = make_raw_loader(test_np, config.batch_size, shuffle=False)
    model, threshold, history = train_torch_model(model=model, train_loader=train_loader, val_loader=val_loader, train_labels=train_np['labels'], optimizer_config=config.optimizer, epochs=config.epochs, patience=config.patience)
    val_labels, val_scores = predict_torch_model(model, val_loader)
    test_labels, test_scores = predict_torch_model(model, test_loader)
    val_metrics = compute_binary_metrics(val_labels, val_scores, threshold)
    test_metrics = compute_binary_metrics(test_labels, test_scores, threshold)
    predictions = _prediction_frame(test_np['metadata'], test_labels, test_scores, threshold)
    sample_accel = torch.from_numpy(test_np['accel'][:64]).float()
    sample_audio = torch.from_numpy(test_np['audio'][:64]).float()
    efficiency = benchmark_torch_model(model, sample_accel, sample_audio)
    _save_confusion_matrix(test_labels, test_scores, threshold, output_dir / 'confusion_matrix.png')
    severity = _severity_analysis(predictions)
    secondary_ds = _secondary_dataset(config)
    secondary_np = dataset_to_numpy(secondary_ds)
    secondary_loader = make_raw_loader(secondary_np, config.batch_size, shuffle=False)
    sec_labels, sec_scores = predict_torch_model(model, secondary_loader)
    secondary_metrics = compute_binary_metrics(sec_labels, sec_scores, threshold)
    secondary_predictions = _prediction_frame(secondary_np['metadata'], sec_labels, sec_scores, threshold)
    secondary_predictions.to_csv(output_dir / 'secondary_predictions.csv', index=False)
    predictions.to_csv(output_dir / 'predictions.csv', index=False)
    (output_dir / 'metrics.json').write_text(json.dumps({'history': history, 'val': val_metrics, 'test': test_metrics, 'secondary': secondary_metrics, 'severity': severity}, indent=2), encoding='utf-8')
    (output_dir / 'efficiency.json').write_text(json.dumps(efficiency, indent=2), encoding='utf-8')
    return {'val': val_metrics, 'test': test_metrics, 'secondary': secondary_metrics, 'severity': severity, 'efficiency': efficiency}

def _train_neural_family(config: ExperimentConfig, output_dir: Path) -> dict[str, Any]:
    train_ds = WindowDataset(config.manifest_path, 'train', config.window_length, config.hop, config.sample_rate, config.crop_seconds, config.use_primary_only)
    val_ds = WindowDataset(config.manifest_path, 'val', config.window_length, config.hop, config.sample_rate, config.crop_seconds, config.use_primary_only)
    test_ds = WindowDataset(config.manifest_path, 'test', config.window_length, config.hop, config.sample_rate, config.crop_seconds, config.use_primary_only)
    if config.feature_frontend == 'mel':
        extractor = MelSpectrogramExtractor(sample_rate=config.sample_rate, n_fft=config.n_fft, hop=config.stft_hop, n_mels=config.n_mels)
        train_x, train_y, _ = _load_or_materialize_tf_dataset(train_ds, extractor, config.modalities, config, 'train')
        val_x, val_y, _ = _load_or_materialize_tf_dataset(val_ds, extractor, config.modalities, config, 'val')
        test_x, test_y, metadata = _load_or_materialize_tf_dataset(test_ds, extractor, config.modalities, config, 'test')
        model = TwoBranchSpectrogramCNN(hidden_dim=config.hidden_dim)
        return _train_torch_feature_model(model, extractor, train_x, train_y, val_x, val_y, test_x, test_y, metadata, config, output_dir)
    if config.feature_frontend == 'wavelet' and config.classifier == 'cnn':
        extractor = WaveletScalogramExtractor(sample_rate=config.sample_rate, n_scales=config.wavelet_scales, output_width=config.wavelet_width, downsample_factor=config.wavelet_downsample)
        train_x, train_y, _ = _load_or_materialize_tf_dataset(train_ds, extractor, config.modalities, config, 'train')
        val_x, val_y, _ = _load_or_materialize_tf_dataset(val_ds, extractor, config.modalities, config, 'val')
        test_x, test_y, metadata = _load_or_materialize_tf_dataset(test_ds, extractor, config.modalities, config, 'test')
        model = TwoBranchSpectrogramCNN(hidden_dim=config.hidden_dim)
        return _train_torch_feature_model(model, extractor, train_x, train_y, val_x, val_y, test_x, test_y, metadata, config, output_dir)
    train_np = dataset_to_numpy(train_ds)
    val_np = dataset_to_numpy(val_ds)
    test_np = dataset_to_numpy(test_ds)
    train_np = _apply_modalities(train_np, config.modalities)
    val_np = _apply_modalities(val_np, config.modalities)
    test_np = _apply_modalities(test_np, config.modalities)
    if config.feature_frontend == 'raw':
        model = RawSignalCNN(hidden_dim=config.hidden_dim)
        return _train_torch_raw_model(model, train_np, val_np, test_np, config, output_dir)
    if config.classifier == 'ann':
        if config.feature_frontend == 'vp_projection':
            model = VPANNModel(n_filters=config.frontend_filters, hidden_dim=config.hidden_dim, vp_latent_dim=config.vp_latent_dim, vp_frame_length=config.vp_frame_length, vp_frame_step=config.vp_frame_step)
        else:
            model = AdaptiveFrontEndANNModel(frontend=config.feature_frontend, n_filters=config.frontend_filters, hidden_dim=config.hidden_dim, vp_latent_dim=config.vp_latent_dim, vp_frame_length=config.vp_frame_length, vp_frame_step=config.vp_frame_step)
        return _train_torch_raw_model(model, train_np, val_np, test_np, config, output_dir)
    if config.feature_frontend == 'vp_projection':
        model = VPSNNModel(encoder=config.encoder, n_filters=config.frontend_filters, hidden_dim=config.snn_hidden_dim, vp_latent_dim=config.vp_latent_dim, vp_frame_length=config.vp_frame_length, vp_frame_step=config.vp_frame_step, vp_encoder_decay=config.vp_encoder_decay, vp_spike_threshold=config.vp_spike_threshold)
    else:
        model = AdaptiveFrontEndSNNModel(frontend=config.feature_frontend, encoder=config.encoder, n_filters=config.frontend_filters, hidden_dim=config.snn_hidden_dim, vp_latent_dim=config.vp_latent_dim, vp_frame_length=config.vp_frame_length, vp_frame_step=config.vp_frame_step, vp_encoder_decay=config.vp_encoder_decay, vp_spike_threshold=config.vp_spike_threshold)
    return _train_torch_raw_model(model, train_np, val_np, test_np, config, output_dir)

def run_experiment(config_path: str | Path) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    _ensure_manifest(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_metrics = []
    for seed in config.seeds:
        set_seed(seed)
        seed_dir = output_dir / f'seed_{seed}'
        seed_dir.mkdir(parents=True, exist_ok=True)
        if config.feature_frontend == 'fft':
            result = _run_classical_fft(config, seed_dir)
        else:
            result = _train_neural_family(config, seed_dir)
        seed_metrics.append(result['test'])
    summary = summarize_metric_runs(seed_metrics)
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return {'summary': summary}
