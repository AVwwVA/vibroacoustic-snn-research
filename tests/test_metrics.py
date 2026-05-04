import numpy as np
from vibro_snn_research.metrics import compute_binary_metrics, select_best_threshold

def test_binary_metrics() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = select_best_threshold(labels, scores)
    metrics = compute_binary_metrics(labels, scores, threshold)
    assert metrics['balanced_accuracy'] >= 0.99
    assert metrics['auroc'] >= 0.99
