from __future__ import annotations
import json
from pathlib import Path
from .plotting import configure_headless_matplotlib
configure_headless_matplotlib()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def aggregate_experiment_outputs(experiments_root: str | Path, output_dir: str | Path) -> tuple[Path, Path, Path]:
    experiments_root = Path(experiments_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for metrics_path in experiments_root.glob('*/seed_*/metrics.json'):
        efficiency_path = metrics_path.parent / 'efficiency.json'
        if not efficiency_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        efficiency = json.loads(efficiency_path.read_text(encoding='utf-8'))
        rows.append({'experiment': metrics_path.parent.parent.name, 'seed': metrics_path.parent.name, 'balanced_accuracy': metrics['test']['balanced_accuracy'], 'auroc': metrics['test']['auroc'], 'secondary_balanced_accuracy': metrics.get('secondary', {}).get('balanced_accuracy'), 'latency_batch': efficiency.get('latency_batch'), 'macs': efficiency.get('macs'), 'parameter_count': efficiency.get('parameter_count'), 'serialized_size_bytes': efficiency.get('serialized_size_bytes'), 'spike_density': efficiency.get('spike_density')})
    frame = pd.DataFrame(rows).sort_values(['experiment', 'seed'])
    table_path = output_dir / 'performance_efficiency_table.csv'
    frame.to_csv(table_path, index=False)
    summary = frame.groupby('experiment', as_index=False).agg(n_seeds=('seed', 'count'), balanced_accuracy_mean=('balanced_accuracy', 'mean'), balanced_accuracy_std=('balanced_accuracy', 'std'), auroc_mean=('auroc', 'mean'), auroc_std=('auroc', 'std'), secondary_balanced_accuracy_mean=('secondary_balanced_accuracy', 'mean'), latency_batch_mean=('latency_batch', 'mean'), macs_mean=('macs', 'mean'), parameter_count_mean=('parameter_count', 'mean'), serialized_size_bytes_mean=('serialized_size_bytes', 'mean'), spike_density_mean=('spike_density', 'mean')).sort_values('balanced_accuracy_mean', ascending=False)
    summary = summary.replace({np.nan: None})
    summary_path = output_dir / 'experiment_summary_table.csv'
    summary.to_csv(summary_path, index=False)
    plt.figure(figsize=(8, 5))
    plt.scatter(summary['latency_batch_mean'], summary['balanced_accuracy_mean'])
    for _, row in summary.iterrows():
        plt.annotate(row['experiment'], (row['latency_batch_mean'], row['balanced_accuracy_mean']))
    plt.xlabel('Latency per batch (s)')
    plt.ylabel('Balanced accuracy')
    plt.title('Performance-efficiency Pareto view')
    plt.tight_layout()
    pareto_path = output_dir / 'pareto_plot.png'
    plt.savefig(pareto_path, dpi=180)
    plt.close()
    return (table_path, summary_path, pareto_path)
