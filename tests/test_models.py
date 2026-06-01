import torch
from vibro_snn_research.models import AdaptiveFrontEndANNModel, AdaptiveFrontEndSNNModel, RawSignalCNN, TwoBranchSpectrogramCNN, VPANNModel, VPProjection1d, VPSNNModel

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

def test_vp_models_and_projection() -> None:
    accel = torch.randn(4, 4096)
    audio = torch.randn(4, 4096)
    projection = VPProjection1d()
    vp_ann = VPANNModel()
    vp_snn = VPSNNModel()
    projected = projection(accel.unsqueeze(1))
    assert projected.shape == (4, 32, 32)
    assert vp_ann(accel, audio).shape == (4,)
    assert vp_snn(accel, audio).shape == (4,)
    assert {'spike_count', 'spike_density', 'estimated_synaptic_ops'} <= vp_snn.last_spike_stats.keys()
