from __future__ import annotations
from pathlib import Path
from .plotting import configure_headless_matplotlib
configure_headless_matplotlib()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .dataset import build_window_index, load_manifest, load_record
from .features import MelSpectrogramExtractor, WaveletScalogramExtractor

def _middle_crop(values: np.ndarray, crop_length: int) -> np.ndarray:
    start = max(0, (len(values) - crop_length) // 2)
    return values[start:start + crop_length]

def _representatives(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {'healthy': frame.loc[frame['health_state'] == 0].iloc[0], 'developing': frame.loc[frame['health_state'] == 1].iloc[0], 'faulty': frame.loc[frame['health_state'] == 2].iloc[0]}

def run_analysis(manifest_path: str | Path, output_dir: str | Path, sample_rate: int=42000, crop_seconds: float=8.0, window_length: int=4096, hop: int=2048) -> dict[str, Path]:
    manifest = load_manifest(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = manifest.loc[manifest['primary_included'] == 1]
    reps = _representatives(primary)
    crop_length = int(sample_rate * crop_seconds)
    plt.figure(figsize=(12, 8))
    for idx, (label, row) in enumerate(reps.items(), start=1):
        record = load_record(row['path'])
        accel = _middle_crop(record['accel'], crop_length)
        plt.subplot(3, 1, idx)
        plt.plot(accel[:8000], linewidth=0.8)
        plt.title(f'Accelerometer waveform: {label}')
        plt.tight_layout()
    waveform_path = output_dir / 'waveforms.png'
    plt.savefig(waveform_path, dpi=200)
    plt.close()
    plt.figure(figsize=(12, 8))
    for idx, (label, row) in enumerate(reps.items(), start=1):
        record = load_record(row['path'])
        accel = _middle_crop(record['accel'], crop_length)
        spectrum = np.log1p(np.abs(np.fft.rfft(accel[:16384])))
        freqs = np.fft.rfftfreq(16384, d=1.0 / sample_rate)
        plt.subplot(3, 1, idx)
        plt.plot(freqs, spectrum, linewidth=0.8)
        plt.title(f'FFT magnitude: {label}')
    fft_path = output_dir / 'fft_overlays.png'
    plt.tight_layout()
    plt.savefig(fft_path, dpi=200)
    plt.close()
    mel_extractor = MelSpectrogramExtractor(sample_rate=sample_rate)
    wavelet_extractor = WaveletScalogramExtractor(sample_rate=sample_rate)
    sample_record = load_record(reps['faulty']['path'])
    sample_window = {'accel': sample_record['accel'][None, :window_length].astype(np.float32), 'audio': sample_record['audio'][None, :window_length].astype(np.float32)}
    mel = mel_extractor.transform(sample_window)[0, 0]
    wavelet = wavelet_extractor.transform(sample_window)[0, 0]
    plt.figure(figsize=(10, 4))
    plt.imshow(mel, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Example log-mel spectrogram (accelerometer)')
    mel_path = output_dir / 'mel_example.png'
    plt.tight_layout()
    plt.savefig(mel_path, dpi=200)
    plt.close()
    plt.figure(figsize=(10, 4))
    plt.imshow(wavelet, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Example wavelet scalogram (accelerometer)')
    wavelet_path = output_dir / 'wavelet_example.png'
    plt.tight_layout()
    plt.savefig(wavelet_path, dpi=200)
    plt.close()
    stability = []
    for _, row in manifest.iterrows():
        record = load_record(row['path'])
        speed = _middle_crop(record['speed'], crop_length)
        load = _middle_crop(record['load'], crop_length)
        stability.append({'record_id': row['record_id'], 'fault_family': row['fault_family'], 'speed_mean': float(speed.mean()), 'speed_std': float(speed.std()), 'load_mean': float(load.mean()), 'load_std': float(load.std())})
    stability_frame = pd.DataFrame(stability)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    stability_frame.boxplot(column='speed_mean', by='fault_family', grid=False)
    plt.title('Speed mean by fault family')
    plt.suptitle('')
    plt.subplot(1, 2, 2)
    stability_frame.boxplot(column='load_mean', by='fault_family', grid=False)
    plt.title('Load mean by fault family')
    stability_path = output_dir / 'speed_load_stability.png'
    plt.tight_layout()
    plt.savefig(stability_path, dpi=200)
    plt.close()
    window_counts = build_window_index(manifest=manifest, window_length=window_length, hop=hop, sample_rate=sample_rate, crop_seconds=crop_seconds, split=None, primary_only=False)
    dataset_card = output_dir / 'dataset_card.md'
    dataset_card.write_text('\n'.join(['# Dataset Card', '', f'- Sample rate: {sample_rate} Hz', f'- Raw duration per recording: 10 s', f'- Cropped duration per recording: {crop_seconds:.1f} s', '- Sensor columns: accelerometer, acoustic, speed, load', '- Primary exclusion: all ball faults and healthy bearings 11-15', '', '## Record counts by split', manifest.groupby(['split', 'subset']).size().to_string(), '', '## Window counts by split', window_counts.groupby('split').size().to_string()]), encoding='utf-8')
    return {'waveforms': waveform_path, 'fft': fft_path, 'mel': mel_path, 'wavelet': wavelet_path, 'stability': stability_path, 'dataset_card': dataset_card}
