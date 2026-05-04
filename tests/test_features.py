import numpy as np
from vibro_snn_research.features import FFTFeatureExtractor, MelSpectrogramExtractor, WaveletScalogramExtractor

def test_feature_extractors_shapes() -> None:
    batch = {'accel': np.random.randn(2, 4096).astype(np.float32), 'audio': np.random.randn(2, 4096).astype(np.float32)}
    fft = FFTFeatureExtractor().transform(batch)
    mel = MelSpectrogramExtractor().transform(batch)
    wavelet = WaveletScalogramExtractor().transform(batch)
    assert fft.shape[0] == 2
    assert mel.shape[:2] == (2, 2)
    assert wavelet.shape[:2] == (2, 2)
