import torch
from vibro_snn_research.models import AdaptiveFrontEndANNModel, AdaptiveFrontEndSNNModel, RawSignalCNN, TwoBranchSpectrogramCNN

def test_model_smoke() -> None:
    accel = torch.randn(4, 4096)
    audio = torch.randn(4, 4096)
    spectrograms = torch.randn(4, 2, 64, 32)
    raw_model = RawSignalCNN()
    ann_model = AdaptiveFrontEndANNModel()
    snn_model = AdaptiveFrontEndSNNModel()
    spec_model = TwoBranchSpectrogramCNN()
    assert raw_model(accel, audio).shape == (4,)
    assert ann_model(accel, audio).shape == (4,)
    assert snn_model(accel, audio).shape == (4,)
    assert spec_model(spectrograms).shape == (4,)
