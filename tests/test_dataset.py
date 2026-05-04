from pathlib import Path
import pandas as pd
from scipy.io import savemat
from vibro_snn_research.dataset import WindowDataset

def test_window_dataset_shapes(tmp_path: Path) -> None:
    path = tmp_path / 'H_1_0.mat'
    savemat(path, {'H_1_0': __import__('numpy').zeros((42000, 4), dtype='float32')})
    manifest = pd.DataFrame([{'record_id': 'H_1_0', 'bearing_id': 1, 'fault_family': 'healthy', 'health_state': 0, 'health_stage': 'healthy', 'binary_label': 0, 'split': 'train', 'subset': 'primary', 'primary_included': 1, 'path': str(path)}])
    dataset = WindowDataset(manifest, split='train', window_length=1024, hop=512, sample_rate=42000, crop_seconds=0.5)
    sample = dataset[0]
    assert sample['accel'].shape[0] == 1024
    assert sample['audio'].shape[0] == 1024
    assert int(sample['label']) == 0
