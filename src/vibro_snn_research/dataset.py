from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset

@lru_cache(maxsize=64)
def load_record(path: str | Path) -> dict[str, np.ndarray]:
    path = str(path)
    mat = loadmat(path)
    key = next((name for name in mat if not name.startswith('__')))
    matrix = np.asarray(mat[key], dtype=np.float32)
    return {'accel': matrix[:, 0].copy(), 'audio': matrix[:, 1].copy(), 'speed': matrix[:, 2].copy(), 'load': matrix[:, 3].copy()}

def load_manifest(manifest: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(manifest, pd.DataFrame):
        return manifest.copy()
    return pd.read_csv(manifest)

def build_window_index(manifest: str | Path | pd.DataFrame, window_length: int, hop: int, sample_rate: int=42000, crop_seconds: float=8.0, split: str | None=None, primary_only: bool=True) -> pd.DataFrame:
    frame = load_manifest(manifest)
    if split is not None:
        frame = frame.loc[frame['split'] == split]
    if primary_only:
        frame = frame.loc[frame['primary_included'] == 1]
    crop_length = int(sample_rate * crop_seconds)
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        signal = load_record(row['path'])['accel']
        total_length = len(signal)
        crop_start = max(0, (total_length - crop_length) // 2)
        crop_end = min(total_length, crop_start + crop_length)
        crop_actual = crop_end - crop_start
        if crop_actual < window_length:
            continue
        window_id = 0
        for start in range(crop_start, crop_end - window_length + 1, hop):
            rows.append({'record_id': row['record_id'], 'bearing_id': int(row['bearing_id']), 'fault_family': row['fault_family'], 'health_state': int(row['health_state']), 'binary_label': int(row['binary_label']), 'split': row['split'], 'subset': row['subset'], 'path': row['path'], 'window_index': window_id, 'start': start, 'stop': start + window_length})
            window_id += 1
    return pd.DataFrame(rows)

class WindowDataset(Dataset):

    def __init__(self, manifest: str | Path | pd.DataFrame, split: str, window_length: int=4096, hop: int=2048, sample_rate: int=42000, crop_seconds: float=8.0, primary_only: bool=True, transform: Any | None=None) -> None:
        self.window_index = build_window_index(manifest=manifest, window_length=window_length, hop=hop, sample_rate=sample_rate, crop_seconds=crop_seconds, split=split, primary_only=primary_only)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.window_index.iloc[idx]
        record = load_record(row['path'])
        accel = record['accel'][row['start']:row['stop']]
        audio = record['audio'][row['start']:row['stop']]
        sample = {'accel': torch.from_numpy(accel.copy()), 'audio': torch.from_numpy(audio.copy()), 'label': torch.tensor(int(row['binary_label']), dtype=torch.long), 'bearing_id': int(row['bearing_id']), 'fault_family': str(row['fault_family']), 'health_state': int(row['health_state']), 'record_id': str(row['record_id']), 'window_index': int(row['window_index'])}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

def dataset_to_numpy(dataset: Dataset) -> dict[str, np.ndarray]:
    accel, audio, labels = ([], [], [])
    metadata: list[dict[str, Any]] = []
    for sample in dataset:
        accel.append(sample['accel'].numpy())
        audio.append(sample['audio'].numpy())
        labels.append(int(sample['label']))
        metadata.append({'bearing_id': sample['bearing_id'], 'fault_family': sample['fault_family'], 'health_state': sample['health_state'], 'record_id': sample['record_id'], 'window_index': sample['window_index']})
    return {'accel': np.stack(accel).astype(np.float32), 'audio': np.stack(audio).astype(np.float32), 'labels': np.asarray(labels, dtype=np.int64), 'metadata': metadata}
