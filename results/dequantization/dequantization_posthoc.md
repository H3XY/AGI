# Dequantization check (POST-HOC sensitivity analysis, wider C grid [1e-05, 3e-05, 0.0001, 0.0003, 0.001, 0.01, 0.1, 1, 10, 100])

Frequency basis: integer vectors in {-2..2}^4, 312 non-zero up to sign (624 cos/sin features).

## Spectrum check (trained seed-0 QNN, 2,000 random inputs)

| Task | Max abs residual, {-2..2} basis | RMS residual | Control: max abs residual, {-1..1} basis |
|---|---|---|---|
| sentiment | 2.89e-15 | 6.94e-16 | 0.319 |
| cve | 2.33e-15 | 6.11e-16 | 0.435 |

## sentiment: confirmatory set (validation), 1066 items

| Model | Seeds 0, 1, 2 | Mean | QNN minus model [two-level 95% CI] |
|---|---|---|---|
| Quantum (QNN) | 0.769, 0.764, 0.765 | 0.766 |  |
| Logistic regression on the 4 features, CV-tuned | 0.769, 0.775, 0.763 | 0.769 |  |
| Fourier surrogate, full spectrum (624 features) | 0.757, 0.765, 0.750 | 0.758 | +0.008 [-0.008, +0.025] |
| Random Fourier subset, k=8 | 0.734, 0.743, 0.697 | 0.725 | +0.042 [+0.013, +0.073] |
| Random Fourier subset, k=32 | 0.725, 0.757, 0.728 | 0.737 | +0.029 [+0.004, +0.055] |
| Random Fourier subset, k=128 | 0.755, 0.769, 0.752 | 0.759 | +0.007 [-0.010, +0.025] |

**Reading (same rule as the pre-registered one, applied post hoc):** inconclusive

## sentiment: exploratory test set, 1066 items

| Model | Seeds 0, 1, 2 | Mean | QNN minus model [two-level 95% CI] |
|---|---|---|---|
| Quantum (QNN) | 0.758, 0.765, 0.748 | 0.757 |  |
| Logistic regression on the 4 features, CV-tuned | 0.774, 0.772, 0.770 | 0.772 |  |
| Fourier surrogate, full spectrum (624 features) | 0.770, 0.769, 0.769 | 0.770 | -0.013 [-0.028, +0.001] |
| Random Fourier subset, k=8 | 0.751, 0.753, 0.689 | 0.731 | +0.025 [-0.003, +0.060] |
| Random Fourier subset, k=32 | 0.737, 0.751, 0.764 | 0.751 | +0.006 [-0.019, +0.028] |
| Random Fourier subset, k=128 | 0.766, 0.764, 0.765 | 0.765 | -0.008 [-0.024, +0.007] |

C grid: [1e-05, 3e-05, 0.0001, 0.0003, 0.001, 0.01, 0.1, 1, 10, 100]; features unscaled (cos/sin in [-1, 1]), L2-regularised logistic regression.
CV-chosen C per seed: {'Fourier surrogate, full spectrum (624 features)': [1e-05, 0.0003, 0.0003], 'Random Fourier subset, k=8': [1, 0.1, 1e-05], 'Random Fourier subset, k=32': [0.001, 0.1, 0.01], 'Random Fourier subset, k=128': [0.0003, 1e-05, 0.0001]}
5-fold CV accuracy at the chosen C, per seed: {'Fourier surrogate, full spectrum (624 features)': [0.7417, 0.7483, 0.7683], 'Random Fourier subset, k=8': [0.7233, 0.7483, 0.7583], 'Random Fourier subset, k=32': [0.7367, 0.735, 0.7583], 'Random Fourier subset, k=128': [0.7333, 0.7467, 0.7767]}

## cve: confirmatory set (holdout), 13432 items

| Model | Seeds 0, 1, 2 | Mean | QNN minus model [two-level 95% CI] |
|---|---|---|---|
| Quantum (QNN) | 0.707, 0.700, 0.718 | 0.708 |  |
| Logistic regression on the 4 features, CV-tuned | 0.720, 0.723, 0.716 | 0.720 |  |
| Fourier surrogate, full spectrum (624 features) | 0.724, 0.730, 0.726 | 0.727 | -0.018 [-0.030, -0.007] |
| Random Fourier subset, k=8 | 0.722, 0.726, 0.727 | 0.725 | -0.017 [-0.026, -0.008] |
| Random Fourier subset, k=32 | 0.719, 0.697, 0.727 | 0.714 | -0.006 [-0.014, +0.004] |
| Random Fourier subset, k=128 | 0.721, 0.727, 0.726 | 0.725 | -0.016 [-0.027, -0.007] |

**Reading (same rule as the pre-registered one, applied post hoc):** classically matched (CI at or below +0.01)

## cve: exploratory test set, 2424 items

| Model | Seeds 0, 1, 2 | Mean | QNN minus model [two-level 95% CI] |
|---|---|---|---|
| Quantum (QNN) | 0.699, 0.680, 0.717 | 0.699 |  |
| Logistic regression on the 4 features, CV-tuned | 0.692, 0.691, 0.683 | 0.689 |  |
| Fourier surrogate, full spectrum (624 features) | 0.700, 0.702, 0.692 | 0.698 | +0.000 [-0.023, +0.024] |
| Random Fourier subset, k=8 | 0.705, 0.697, 0.700 | 0.700 | -0.002 [-0.022, +0.018] |
| Random Fourier subset, k=32 | 0.704, 0.666, 0.695 | 0.688 | +0.010 [-0.007, +0.027] |
| Random Fourier subset, k=128 | 0.703, 0.703, 0.698 | 0.701 | -0.002 [-0.024, +0.020] |

C grid: [1e-05, 3e-05, 0.0001, 0.0003, 0.001, 0.01, 0.1, 1, 10, 100]; features unscaled (cos/sin in [-1, 1]), L2-regularised logistic regression.
CV-chosen C per seed: {'Fourier surrogate, full spectrum (624 features)': [0.001, 0.0003, 0.001], 'Random Fourier subset, k=8': [100, 1e-05, 1e-05], 'Random Fourier subset, k=32': [0.01, 10, 0.01], 'Random Fourier subset, k=128': [0.001, 0.0003, 0.0001]}
5-fold CV accuracy at the chosen C, per seed: {'Fourier surrogate, full spectrum (624 features)': [0.7483, 0.745, 0.755], 'Random Fourier subset, k=8': [0.7217, 0.7417, 0.7533], 'Random Fourier subset, k=32': [0.735, 0.7417, 0.7567], 'Random Fourier subset, k=128': [0.7417, 0.7367, 0.765]}
