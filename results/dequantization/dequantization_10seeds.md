# Dequantization check (10-seed extension, see results/confirmatory/EXTENSION_10SEEDS.md; primary reading is the 3-seed one)

Frequency basis: integer vectors in {-2..2}^4, 312 non-zero up to sign (624 cos/sin features).

## Spectrum check (trained seed-0 QNN, 2,000 random inputs)

| Task | Max abs residual, {-2..2} basis | RMS residual | Control: max abs residual, {-1..1} basis |
|---|---|---|---|
| sentiment | 2.89e-15 | 6.94e-16 | 0.319 |
| cve | 2.33e-15 | 6.11e-16 | 0.435 |

## sentiment: confirmatory set (validation), 1066 items

| Model | Seeds 0-9 | Mean | QNN minus model [two-level 95% CI] |
|---|---|---|---|
| Quantum (QNN) | 0.769, 0.764, 0.765, 0.765, 0.771, 0.759, 0.765, 0.762, 0.763, 0.774 | 0.766 |  |
| Logistic regression on the 4 features, CV-tuned | 0.769, 0.775, 0.763, 0.749, 0.765, 0.760, 0.765, 0.750, 0.773, 0.780 | 0.765 |  |
| Fourier surrogate, full spectrum (624 features) | 0.758, 0.766, 0.743, 0.743, 0.764, 0.743, 0.750, 0.747, 0.748, 0.768 | 0.753 | +0.013 [-0.000, +0.026] |
| Random Fourier subset, k=8 | 0.734, 0.743, 0.741, 0.733, 0.742, 0.725, 0.741, 0.749, 0.749, 0.750 | 0.741 | +0.025 [+0.008, +0.042] |
| Random Fourier subset, k=32 | 0.725, 0.757, 0.728, 0.737, 0.755, 0.735, 0.738, 0.749, 0.747, 0.759 | 0.743 | +0.023 [+0.008, +0.038] |
| Random Fourier subset, k=128 | 0.747, 0.766, 0.744, 0.746, 0.765, 0.749, 0.743, 0.746, 0.748, 0.761 | 0.751 | +0.014 [+0.002, +0.027] |

**Same reading rule, 10 seeds:** inconclusive

## sentiment: exploratory test set, 1066 items

| Model | Seeds 0-9 | Mean | QNN minus model [two-level 95% CI] |
|---|---|---|---|
| Quantum (QNN) | 0.758, 0.765, 0.748, 0.757, 0.749, 0.757, 0.750, 0.750, 0.763, 0.749 | 0.754 |  |
| Logistic regression on the 4 features, CV-tuned | 0.774, 0.772, 0.770, 0.770, 0.771, 0.774, 0.775, 0.771, 0.767, 0.769 | 0.771 |  |
| Fourier surrogate, full spectrum (624 features) | 0.771, 0.770, 0.767, 0.760, 0.770, 0.755, 0.759, 0.765, 0.764, 0.765 | 0.765 | -0.010 [-0.022, +0.001] |
| Random Fourier subset, k=8 | 0.751, 0.753, 0.756, 0.747, 0.766, 0.752, 0.761, 0.764, 0.756, 0.750 | 0.756 | -0.001 [-0.017, +0.014] |
| Random Fourier subset, k=32 | 0.737, 0.751, 0.764, 0.752, 0.762, 0.754, 0.755, 0.759, 0.762, 0.758 | 0.755 | -0.001 [-0.015, +0.013] |
| Random Fourier subset, k=128 | 0.766, 0.769, 0.766, 0.746, 0.767, 0.756, 0.743, 0.765, 0.759, 0.754 | 0.759 | -0.005 [-0.017, +0.007] |

C grid: [0.001, 0.01, 0.1, 1, 10, 100]; features unscaled (cos/sin in [-1, 1]), L2-regularised logistic regression.
CV-chosen C per seed: {'Fourier surrogate, full spectrum (624 features)': [0.001, 0.001, 0.001, 0.001, 0.001, 0.01, 0.01, 0.001, 0.001, 0.001], 'Random Fourier subset, k=8': [1, 0.1, 10, 1, 0.1, 0.1, 100, 10, 1, 0.1], 'Random Fourier subset, k=32': [0.001, 0.1, 0.01, 0.1, 0.1, 0.1, 0.1, 1, 0.1, 0.1], 'Random Fourier subset, k=128': [0.001, 0.001, 0.001, 0.1, 0.001, 0.001, 0.1, 0.001, 0.001, 0.001]}
5-fold CV accuracy at the chosen C, per seed: {'Fourier surrogate, full spectrum (624 features)': [0.7217, 0.7467, 0.7667, 0.745, 0.7633, 0.7617, 0.7417, 0.765, 0.795, 0.7483], 'Random Fourier subset, k=8': [0.7233, 0.7483, 0.7567, 0.7367, 0.74, 0.77, 0.7467, 0.7367, 0.77, 0.7533], 'Random Fourier subset, k=32': [0.7367, 0.735, 0.7583, 0.755, 0.75, 0.7583, 0.7417, 0.7467, 0.78, 0.7533], 'Random Fourier subset, k=128': [0.73, 0.7467, 0.7717, 0.7533, 0.7583, 0.7633, 0.74, 0.7633, 0.7933, 0.7483]}

## cve: confirmatory set (holdout), 13432 items

| Model | Seeds 0-9 | Mean | QNN minus model [two-level 95% CI] |
|---|---|---|---|
| Quantum (QNN) | 0.707, 0.700, 0.718, 0.719, 0.714, 0.714, 0.706, 0.715, 0.704, 0.716 | 0.711 |  |
| Logistic regression on the 4 features, CV-tuned | 0.720, 0.723, 0.716, 0.719, 0.720, 0.720, 0.726, 0.717, 0.722, 0.722 | 0.720 |  |
| Fourier surrogate, full spectrum (624 features) | 0.724, 0.730, 0.726, 0.728, 0.718, 0.732, 0.728, 0.725, 0.728, 0.728 | 0.727 | -0.015 [-0.022, -0.010] |
| Random Fourier subset, k=8 | 0.722, 0.725, 0.722, 0.719, 0.727, 0.725, 0.724, 0.722, 0.726, 0.728 | 0.724 | -0.013 [-0.019, -0.007] |
| Random Fourier subset, k=32 | 0.719, 0.697, 0.727, 0.713, 0.727, 0.731, 0.725, 0.719, 0.727, 0.727 | 0.721 | -0.010 [-0.016, -0.003] |
| Random Fourier subset, k=128 | 0.721, 0.729, 0.718, 0.726, 0.728, 0.730, 0.726, 0.723, 0.727, 0.727 | 0.725 | -0.014 [-0.020, -0.008] |

**Same reading rule, 10 seeds:** classically matched (CI at or below +0.01)

## cve: exploratory test set, 2424 items

| Model | Seeds 0-9 | Mean | QNN minus model [two-level 95% CI] |
|---|---|---|---|
| Quantum (QNN) | 0.699, 0.680, 0.717, 0.682, 0.705, 0.708, 0.705, 0.703, 0.697, 0.708 | 0.700 |  |
| Logistic regression on the 4 features, CV-tuned | 0.692, 0.691, 0.683, 0.693, 0.682, 0.696, 0.697, 0.678, 0.687, 0.696 | 0.690 |  |
| Fourier surrogate, full spectrum (624 features) | 0.700, 0.698, 0.692, 0.697, 0.687, 0.705, 0.715, 0.703, 0.693, 0.695 | 0.698 | +0.002 [-0.010, +0.013] |
| Random Fourier subset, k=8 | 0.705, 0.698, 0.694, 0.688, 0.702, 0.712, 0.707, 0.701, 0.696, 0.699 | 0.700 | +0.000 [-0.011, +0.011] |
| Random Fourier subset, k=32 | 0.704, 0.666, 0.695, 0.689, 0.692, 0.706, 0.717, 0.705, 0.693, 0.700 | 0.697 | +0.004 [-0.006, +0.014] |
| Random Fourier subset, k=128 | 0.703, 0.697, 0.690, 0.692, 0.688, 0.701, 0.713, 0.700, 0.693, 0.697 | 0.697 | +0.003 [-0.008, +0.014] |

C grid: [0.001, 0.01, 0.1, 1, 10, 100]; features unscaled (cos/sin in [-1, 1]), L2-regularised logistic regression.
CV-chosen C per seed: {'Fourier surrogate, full spectrum (624 features)': [0.001, 0.001, 0.001, 0.001, 0.1, 0.001, 0.001, 0.001, 0.001, 0.001], 'Random Fourier subset, k=8': [100, 0.001, 0.1, 1, 0.01, 0.001, 0.001, 0.01, 0.01, 0.001], 'Random Fourier subset, k=32': [0.01, 10, 0.01, 10, 0.01, 0.1, 0.01, 0.01, 0.01, 0.01], 'Random Fourier subset, k=128': [0.001, 0.001, 0.01, 0.01, 0.001, 0.01, 0.01, 0.01, 0.001, 0.001]}
5-fold CV accuracy at the chosen C, per seed: {'Fourier surrogate, full spectrum (624 features)': [0.7483, 0.7417, 0.755, 0.7367, 0.7717, 0.755, 0.75, 0.7617, 0.7517, 0.7667], 'Random Fourier subset, k=8': [0.7217, 0.74, 0.7467, 0.7317, 0.7633, 0.7533, 0.76, 0.7717, 0.7583, 0.7667], 'Random Fourier subset, k=32': [0.735, 0.7417, 0.7567, 0.7467, 0.7733, 0.7417, 0.7483, 0.76, 0.75, 0.765], 'Random Fourier subset, k=128': [0.7417, 0.7333, 0.7583, 0.735, 0.7817, 0.7567, 0.7467, 0.765, 0.76, 0.77]}
