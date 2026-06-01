from pathlib import Path
from vibro_snn_research.config import load_experiment_config

def test_vp_configs_load() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    vp_ann = load_experiment_config(repo_root / 'configs' / 'vp_ann.yaml')
    vp_snn = load_experiment_config(repo_root / 'configs' / 'vp_snn.yaml')
    filterbank = load_experiment_config(repo_root / 'configs' / 'filterbank_ann.yaml')
    assert vp_ann.feature_frontend == 'vp_projection'
    assert vp_ann.classifier == 'ann'
    assert vp_ann.vp_latent_dim == 32
    assert vp_ann.vp_frame_length == 128
    assert vp_snn.feature_frontend == 'vp_projection'
    assert vp_snn.classifier == 'snn'
    assert vp_snn.vp_encoder_decay == 0.9
    assert vp_snn.vp_spike_threshold == 0.02
    assert filterbank.feature_frontend == 'learnable_filterbank'
