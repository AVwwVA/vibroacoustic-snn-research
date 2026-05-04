# Vibroacoustic SNN Research Pipeline

This project implements a reproducible vibroacoustic fault and anomaly detection
pipeline for the **University of Ottawa Ball-bearing Vibration and Acoustic Fault
Data under Constant Load and Speed Conditions (UODS-VAFDC)** dataset.

## Implemented scope

- Ottawa dataset extraction, manifest creation, and leakage-safe bearing splits
- Windowed dataset with synchronized accelerometer and acoustic channels
- EDA pipeline with waveform, FFT, mel, and wavelet plots plus dataset card
- FFT/XGBoost and FFT/MLP baselines
- Fixed log-mel CNN and fixed wavelet CNN baselines
- Raw-signal 1D CNN baseline
- Learnable filterbank ANN
- Adaptive wavelet ANN
- Low-power adaptive-front-end + LIF SNN with delta or rate encoding
- Efficiency reporting: parameters, model size, latency, memory, FLOPs/MACs,
  activation density, and SNN spike statistics

## Quickstart

```bash
python3 -m venv .venv
./.venv/bin/pip install -e .
./.venv/bin/python scripts/build_manifest.py
./.venv/bin/python scripts/run_analysis.py --manifest outputs/manifests/ottawa_manifest.csv
./.venv/bin/python scripts/run_experiment.py --config configs/fft_xgboost.yaml
```

## Recommended validation order

Use the smoke configs first to validate both neural paths on the real Ottawa data:

```bash
./.venv/bin/python scripts/run_experiment.py --config configs/mel_cnn_smoke.yaml
./.venv/bin/python scripts/run_experiment.py --config configs/filterbank_snn_smoke.yaml
./.venv/bin/python scripts/aggregate_results.py --experiments-root outputs/experiments_smoke --output-dir outputs/reports_smoke
```

Then move on to the full research configs in `configs/`.

## Notes

- The plotting scripts automatically use a project-local matplotlib cache and the
  `Agg` backend, so they work in headless environments.
- If `xgboost` cannot load its native OpenMP runtime on macOS, the FFT boosted
  tree baseline falls back to a scikit-learn histogram gradient booster with the
  same training and evaluation flow.

## Dataset assumptions

- The first matrix column is accelerometer data.
- The second matrix column is acoustic data.
- The third matrix column is speed.
- The fourth matrix column is load.
- All data is sampled at **42,000 Hz** for **10 seconds**.
- The primary study crops the middle **8 seconds** and excludes **ball faults**
  and healthy bearings `11-15` from the primary experiment set.
