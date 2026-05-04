from vibro_snn_research.manifest import build_manifest, parse_record_name

def test_parse_record_name_primary_inner_fault() -> None:
    meta = parse_record_name('I_2_1.mat')
    assert meta.record_id == 'I_2_1'
    assert meta.bearing_id == 2
    assert meta.fault_family == 'inner_race'
    assert meta.health_state == 1
    assert meta.binary_label == 1
    assert meta.split == 'train'
    assert meta.primary_included is True

def test_parse_record_name_secondary_ball_fault() -> None:
    meta = parse_record_name('B_11_2.mat')
    assert meta.fault_family == 'ball'
    assert meta.split == 'secondary'
    assert meta.primary_included is False
