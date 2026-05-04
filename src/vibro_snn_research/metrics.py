from __future__ import annotations
from typing import Iterable
import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

def select_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        score = balanced_accuracy_score(y_true, (y_score >= threshold).astype(int))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold

def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / max(tn + fp, 1)
    sensitivity = tp / max(tp + fn, 1)
    return {'threshold': float(threshold), 'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)), 'auroc': float(roc_auc_score(y_true, y_score)), 'auprc': float(average_precision_score(y_true, y_score)), 'f1': float(f1_score(y_true, y_pred)), 'sensitivity': float(sensitivity), 'specificity': float(specificity), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

def summarize_metric_runs(metrics: Iterable[dict[str, float]]) -> dict[str, dict[str, float]]:
    rows = list(metrics)
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = {'mean': float(values.mean()), 'std': float(values.std(ddof=0))}
    return summary
