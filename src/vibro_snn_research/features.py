from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import numpy as np
import pywt
from scipy import signal, stats

class FeatureExtractor(ABC):

    @abstractmethod
    def fit(self, train_manifest: Any | None=None) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def transform(self, window_batch: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

def _time_stats(values: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(values ** 2, axis=1))
    variance = np.var(values, axis=1)
    kurt = stats.kurtosis(values, axis=1, fisher=False, bias=False)
    crest = np.max(np.abs(values), axis=1) / np.clip(rms, 1e-08, None)
    peak_to_peak = np.ptp(values, axis=1)
    return np.stack([rms, variance, kurt, crest, peak_to_peak], axis=1)

def _spectral_stats(values: np.ndarray, sample_rate: int) -> np.ndarray:
    magnitudes = np.abs(np.fft.rfft(values, axis=1))
    freqs = np.fft.rfftfreq(values.shape[1], d=1.0 / sample_rate)
    centroid = (magnitudes * freqs[None, :]).sum(axis=1) / np.clip(magnitudes.sum(axis=1), 1e-08, None)
    rolloff = np.empty(values.shape[0], dtype=np.float32)
    for idx, mag in enumerate(magnitudes):
        cumsum = np.cumsum(mag)
        threshold = 0.85 * cumsum[-1]
        rolloff[idx] = freqs[np.searchsorted(cumsum, threshold)]
    flatness = stats.gmean(np.clip(magnitudes, 1e-08, None), axis=1) / np.clip(magnitudes.mean(axis=1), 1e-08, None)
    return np.stack([centroid, rolloff, flatness], axis=1).astype(np.float32)

@dataclass(slots=True)
class FFTFeatureExtractor(FeatureExtractor):
    sample_rate: int = 42000
    n_bands: int = 256

    def fit(self, train_manifest: Any | None=None) -> dict[str, Any]:
        return {'frontend': 'fft', 'sample_rate': self.sample_rate, 'n_bands': self.n_bands}

    def transform(self, window_batch: dict[str, np.ndarray]) -> np.ndarray:
        feature_blocks = []
        for modality in ('accel', 'audio'):
            values = window_batch[modality]
            spectrum = np.abs(np.fft.rfft(values, axis=1))
            spectrum = np.log1p(spectrum)
            pooled = np.array_split(spectrum, self.n_bands, axis=1)
            pooled = np.stack([chunk.mean(axis=1) for chunk in pooled], axis=1)
            feature_blocks.append(pooled.astype(np.float32))
            feature_blocks.append(_time_stats(values).astype(np.float32))
            feature_blocks.append(_spectral_stats(values, self.sample_rate))
        return np.concatenate(feature_blocks, axis=1)

def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:

    def hz_to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mels: np.ndarray) -> np.ndarray:
        return 700.0 * (10 ** (mels / 2595.0) - 1.0)
    min_mel = hz_to_mel(np.array([0.0]))[0]
    max_mel = hz_to_mel(np.array([sample_rate / 2]))[0]
    mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for idx in range(1, n_mels + 1):
        left, center, right = (bins[idx - 1], bins[idx], bins[idx + 1])
        center = max(center, left + 1)
        right = max(right, center + 1)
        center = min(center, filters.shape[1] - 1)
        right = min(right, filters.shape[1])
        if right <= center or center <= left:
            continue
        filters[idx - 1, left:center] = np.linspace(0, 1, center - left, endpoint=False)
        filters[idx - 1, center:right] = np.linspace(1, 0, right - center, endpoint=False)
    return filters

@dataclass(slots=True)
class MelSpectrogramExtractor(FeatureExtractor):
    sample_rate: int = 42000
    n_fft: int = 512
    hop: int = 128
    n_mels: int = 64

    def fit(self, train_manifest: Any | None=None) -> dict[str, Any]:
        return {'frontend': 'mel', 'sample_rate': self.sample_rate, 'n_fft': self.n_fft, 'hop': self.hop, 'n_mels': self.n_mels}

    def transform(self, window_batch: dict[str, np.ndarray]) -> np.ndarray:
        bank = mel_filterbank(self.sample_rate, self.n_fft, self.n_mels)
        outputs = []
        for modality in ('accel', 'audio'):
            channel_specs = []
            for signal_batch in window_batch[modality]:
                _, _, stft = signal.stft(signal_batch, fs=self.sample_rate, nperseg=self.n_fft, noverlap=self.n_fft - self.hop, boundary=None)
                power = np.abs(stft) ** 2
                mel_spec = np.log1p(bank @ power)
                channel_specs.append(mel_spec.astype(np.float32))
            outputs.append(np.stack(channel_specs, axis=0))
        return np.stack(outputs, axis=1)

@dataclass(slots=True)
class WaveletScalogramExtractor(FeatureExtractor):
    sample_rate: int = 42000
    n_scales: int = 64
    output_width: int = 128
    wavelet: str = 'morl'
    downsample_factor: int = 4

    def fit(self, train_manifest: Any | None=None) -> dict[str, Any]:
        return {'frontend': 'wavelet', 'sample_rate': self.sample_rate, 'n_scales': self.n_scales, 'output_width': self.output_width, 'wavelet': self.wavelet, 'downsample_factor': self.downsample_factor}

    def transform(self, window_batch: dict[str, np.ndarray]) -> np.ndarray:
        scales = np.arange(1, self.n_scales + 1)
        outputs = []
        for modality in ('accel', 'audio'):
            channel_specs = []
            for signal_batch in window_batch[modality]:
                signal_view = signal_batch
                sample_rate = self.sample_rate
                if self.downsample_factor > 1:
                    signal_view = signal_batch[::self.downsample_factor]
                    sample_rate = self.sample_rate / self.downsample_factor
                coeffs, _ = pywt.cwt(signal_view, scales, self.wavelet, sampling_period=1.0 / sample_rate)
                coeffs = np.abs(coeffs)
                original_x = np.linspace(0.0, 1.0, coeffs.shape[1], dtype=np.float32)
                target_x = np.linspace(0.0, 1.0, self.output_width, dtype=np.float32)
                resized = np.vstack([np.interp(target_x, original_x, row) for row in coeffs]).astype(np.float32)
                channel_specs.append(resized)
            outputs.append(np.stack(channel_specs, axis=0))
        return np.stack(outputs, axis=1)
