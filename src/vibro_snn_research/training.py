from __future__ import annotations
import copy
import dataclasses
import json
import random
from typing import Any
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
from .config import OptimizerConfig
from .metrics import compute_binary_metrics, select_best_threshold

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def make_tensor_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(features).float(), torch.from_numpy(labels).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def make_raw_loader(features: dict[str, np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(features['accel']).float(), torch.from_numpy(features['audio']).float(), torch.from_numpy(features['labels']).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_classical_model(classifier_name: str, train_features: np.ndarray, train_labels: np.ndarray, val_features: np.ndarray, val_labels: np.ndarray, seed: int) -> tuple[Any, float]:
    if classifier_name == 'xgboost':
        if XGBClassifier is None:
            model = Pipeline([('scale', StandardScaler()), ('hist_gb', HistGradientBoostingClassifier(learning_rate=0.05, max_depth=4, max_iter=200, random_state=seed))])
        else:
            model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric='logloss', random_state=seed)
        model.fit(train_features, train_labels)
        val_scores = model.predict_proba(val_features)[:, 1]
        threshold = select_best_threshold(val_labels, val_scores)
        return (model, threshold)
    model = Pipeline([('scale', StandardScaler()), ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', alpha=0.0001, batch_size=128, learning_rate_init=0.001, max_iter=300, random_state=seed))])
    model.fit(train_features, train_labels)
    val_scores = model.predict_proba(val_features)[:, 1]
    threshold = select_best_threshold(val_labels, val_scores)
    return (model, threshold)

def predict_classical_model(model: Any, features: np.ndarray) -> np.ndarray:
    return model.predict_proba(features)[:, 1]

def _build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    if config.name.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, momentum=0.9)
    return torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

def _binary_loss(labels: np.ndarray) -> torch.Tensor | None:
    positives = float(labels.sum())
    negatives = float(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return None
    return torch.tensor(negatives / positives, dtype=torch.float32)

def train_torch_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, train_labels: np.ndarray, optimizer_config: OptimizerConfig, epochs: int, patience: int, device: str='cpu') -> tuple[nn.Module, float, list[dict[str, float]]]:
    model = model.to(device)
    optimizer = _build_optimizer(model, optimizer_config)
    pos_weight = _binary_loss(train_labels)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_state = copy.deepcopy(model.state_dict())
    best_score = -1.0
    best_threshold = 0.5
    history: list[dict[str, float]] = []
    stale = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            if len(batch) == 3:
                accel, audio, labels = batch
                logits = model(accel.to(device), audio.to(device))
            else:
                features, labels = batch
                logits = model(features.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        val_labels_array, val_scores = predict_torch_model(model, val_loader, device=device)
        threshold = select_best_threshold(val_labels_array, val_scores)
        metrics = compute_binary_metrics(val_labels_array, val_scores, threshold)
        metrics['epoch'] = float(epoch)
        metrics['loss'] = running_loss / max(len(train_loader), 1)
        history.append(metrics)
        if metrics['balanced_accuracy'] > best_score:
            best_score = metrics['balanced_accuracy']
            best_threshold = threshold
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return (model, best_threshold, history)

def predict_torch_model(model: nn.Module, loader: DataLoader, device: str='cpu') -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, scores = ([], [])
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                accel, audio, batch_labels = batch
                logits = model(accel.to(device), audio.to(device))
            else:
                features, batch_labels = batch
                logits = model(features.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            scores.extend(probs.tolist())
            labels.extend(batch_labels.numpy().astype(int).tolist())
    return (np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float32))
