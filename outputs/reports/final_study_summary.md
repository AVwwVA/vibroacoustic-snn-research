# Final Study Summary

## Study setup

- Dataset: University of Ottawa Ball-bearing Vibration and Acoustic Fault Data under Constant Load and Speed Conditions (UODS-VAFDC)
- Task: binary classification (`healthy` vs `abnormal`)
- Inputs: accelerometer + acoustic channels
- Primary split: bearing-level train/val/test split
- Secondary evaluation: held-out load-shift subset (ball faults and healthy bearings 11-15)
- Windowing: 4096-sample windows, 2048-sample hop, middle 8 seconds of each recording

## Dataset size

- Primary train windows: 4401
- Primary validation windows: 1467
- Primary test windows: 1467
- Secondary windows: 2445

## Models compared

1. FFT + XGBoost
2. FFT + MLP
3. Log-mel CNN
4. Wavelet CNN
5. Raw-signal CNN
6. Learnable filterbank ANN
7. Learnable filterbank SNN
8. Adaptive wavelet ANN
9. Adaptive wavelet SNN

## Main results

| Model | Seeds | Balanced Acc. | AUROC | Secondary Balanced Acc. | Latency / batch | Params | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Wavelet CNN | 1 | 0.9893 | 1.0000 | 0.8497 | 0.1002 s | 135,777 | Best raw test accuracy, but highest cost and only 1 completed seed |
| Log-mel CNN | 5 | 0.9658 | 1.0000 | 0.8067 | 0.0230 s | 135,777 | Strongest overall conventional model |
| FFT + XGBoost | 1 | 0.9213 | 0.9974 | 0.8745 | 0.0046 s | n/a | Best classical baseline |
| FFT + MLP | 1 | 0.9167 | 1.0000 | 0.9000 | 0.00019 s | 35,969 | Fastest baseline, excellent robustness |
| Learnable filterbank ANN | 2 | 0.9070 | 0.9636 | 0.7945 | 0.0697 s | 139,457 | Best learned raw front-end ANN |
| Learnable filterbank SNN | 1 | 0.8681 | 0.9144 | 0.5117 | 0.0741 s | 24,897 | Best spiking model in this study |
| Raw-signal CNN | 5 | 0.8663 | 0.8357 | 0.6785 | 0.0272 s | 111,297 | Weaker and less stable than mel-CNN |
| Adaptive wavelet SNN | 1 | 0.7198 | 0.7631 | 0.5031 | 0.0797 s | 16,833 | Too insensitive to abnormalities |
| Adaptive wavelet ANN | 1 | 0.4954 | 0.8337 | 0.5000 | 0.0726 s | 131,393 | Collapsed toward predicting almost everything as abnormal |

## Interpretation

### 1. Best conventional model

The **log-mel CNN** is the strongest model to lead with in the thesis.

Why:
- best multi-seed result among the fully repeated neural models
- perfect mean AUROC on the held-out test set
- far more stable than the raw-signal CNN
- much cheaper than the wavelet CNN

### 2. Best low-power / SNN model

The **learnable filterbank SNN** is the best spiking result and should be the main SNN headline.

Why:
- much smaller than the CNN baselines (`24,897` parameters vs `135,777` for mel-CNN)
- serialized size only about `103 KB`
- better than the adaptive-wavelet SNN
- competitive with the raw-signal CNN on primary-test balanced accuracy

Limitation:
- weak secondary-set robustness under load shift (`0.5117` secondary balanced accuracy)

### 3. Classical baseline conclusion

The **FFT baselines are strong**.

- FFT + XGBoost is a very good accuracy baseline
- FFT + MLP is extremely fast and surprisingly robust on the secondary split

This means the thesis should not claim that neural models always dominate traditional signal processing.  
The more defensible claim is that **time-frequency neural models improve peak accuracy**, while **classical FFT baselines remain excellent low-cost comparators**.

### 4. Adaptive wavelet conclusion

The adaptive-wavelet models did **not** justify becoming the main contribution in the current implementation.

- the adaptive-wavelet ANN failed badly
- the adaptive-wavelet SNN underperformed the filterbank SNN

So adaptive wavelets should be presented as an exploratory extension, not the core result.

## Recommended thesis narrative

Use this framing:

1. **Problem**: detect machine faults from vibroacoustic signals.
2. **Baselines**: compare FFT and fixed time-frequency models.
3. **Main result**: log-mel CNN gives the best conventional accuracy.
4. **Low-power result**: learnable filterbank SNN offers a compact spiking alternative with much smaller model size.
5. **Negative result**: adaptive wavelet front-ends were not competitive enough in the current setup.

## Final recommendation

For the final thesis/report, present:

- **Best accuracy model**: `mel_cnn`
- **Best classical baseline**: `fft_xgboost` and `fft_mlp`
- **Best SNN model**: `filterbank_snn`
- **Extension / exploratory model**: `wavelet_cnn`
- **Do not lead with**: `adaptive_wavelet_ann`, `adaptive_wavelet_snn`

## Practical next step

If this work is extended later, the best next improvement is:

- improve the SNN front-end and encoding so that secondary robustness increases
- keep the mel-CNN as the accuracy target to beat
